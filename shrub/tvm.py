import tvm
import os
import logging
import numpy as np
from tvm import autotvm, relay
import tvm.contrib.graph_runtime as runtime
from tvm.autotvm.tuner import XGBTuner

logger = logging.getLogger('tuning')


class TargetProvider:
    class Tracker:
        def __init__(self, ip='0.0.0.0', port=9190):
            self.ip = ip
            self.port = port

        def asTuple(self):
            return (self.ip, self.port)

    class RpcInfo:
        def __init__(self, key, path):
            self.key = key
            self.path = path

    class ThreadMod:
        def __init__(self, type, num):
            self.type = type
            self.num = num

        def asTuple(self):
            return (self.type, self.num)

    def __init__(self, key, core_type=1, core_num=1):
        self.tracker = self.Tracker()
        self.core = self.ThreadMod(core_type, core_num)
        self.ndk_opts = [
            '-shared',
            '-fPIC',
        ]
        self._get_target(key)

    def _get_target(self, key):
        arm_device = 'llvm -device=arm_cpu -mattr=+neon'
        aarch64_target = '-target=aarch64-none-linux-gnueabi'

        if key in ['rasp', 'dmlc']:
            self.target = arm_device + ' ' + aarch64_target + ' -mcpu=cortex-a53'
            self.host = self.target
            self.rpc = self.RpcInfo(key, '/home/wzh/rpc.tvm/')
            os.environ['TVM_NDK_CC'] = 'aarch64-linux-gnu-g++'
        if key in [
                'cpu',
        ]:
            self.target = 'llvm -target=x86_64-none-linux-gnueabi'
            self.host = self.target
            self.rpc = self.RpcInfo(key,
                                    '/home/scratch.zhenhuaw_sw/tvm/rpc/cpu/')
            os.environ['TVM_NDK_CC'] = 'g++'
        elif key == 'llvm':
            self.target = key
            self.host = self.target
            self.rpc = self.RpcInfo(key, None)
        else:
            raise ValueError("Unknow target name %s", key)


class Deployable:
    def build(self, target, model, net, params, record=None, dumpIR=False):
        logging.info("Compiling...")
        if record and record.pick_best():
            with autotvm.apply_history_best(record.best):
                self.graph, self.lib, self.params = relay.build(
                    net,
                    target=target.target,
                    target_host=target.host,
                    params=params)
        else:
            self.graph, self.lib, self.params = relay.build(
                net,
                target=target.target,
                target_host=target.host,
                params=params)

    def export(self, target, export_path=None):
        if export_path is None:
            export_path = os.path.join(target.rpc.key, 'deploy')
        logging.info("Exporting deployables to %s..." % export_path)

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        self.path_so = os.path.join(export_path, 'lib.so')
        if target.rpc.path:
            from tvm.contrib import ndk
            self.lib.export_library(self.path_so,
                                    ndk.create_shared,
                                    options=target.ndk_opts)
        else:
            self.lib.export_library(self.path_so)
        if 'llvm' in target.target:
            self.lib.save(os.path.join(export_path, 'lib.ll'), 'll')
            self.lib.save(os.path.join(export_path, 'lib.asm'), 'asm')
            self.lib.save(os.path.join(export_path, 'lib.o'), 'o')
        elif 'opencl' in target.target:
            self.lib.imported_modules[0].save(
                os.path.join(export_path, 'lib.cl'), 'cl')

        path_params = os.path.join(export_path, 'param.params')
        with open(path_params, 'wb') as fo:
            fo.write(relay.save_param_dict(self.params))


class RuntimeWrapper:
    def __init__(self, model, target, deployables):
        if target.rpc.path:
            # upload module to device
            logging.info("Upload...")
            remote = autotvm.measure.request_remote(target.rpc.key,
                                                    target.tracker.ip,
                                                    target.tracker.port,
                                                    timeout=1000)
            remote.upload(deployables.path_so,
                          target=target.rpc.path + 'mylib.so')
            logging.info("Load library...")
            rlib = remote.load_module(target.rpc.path + 'mylib.so')

            my_config_threadpool = remote.get_function(
                'runtime.config_threadpool')
            my_config_threadpool(target.core.type, target.core.num)
            rparams = {
                k: tvm.nd.array(v, remote.context(str(target.target), 0))
                for k, v in deployables.params.items()
            }
            # upload parameters to device
            ctx = remote.context(str(target.target), 0)
            module = runtime.create(deployables.graph, rlib, ctx)
            for i in range(0, len(model.inputs)):
                module.set_input(model.inputs[i].name, model.inputs[i].ndarray)
            module.set_input(**rparams)
        else:
            ctx = tvm.context(str(target.target), 0)
            rlib = tvm.runtime.load_module(deployables.path_so)
            module = runtime.create(deployables.graph, rlib, ctx)
            for i in range(0, len(model.inputs)):
                module.set_input(model.inputs[i].name,
                                 tvm.nd.array(model.inputs[i].ndarray))
            module.set_input(**deployables.params)
        self.module = module
        self.ctx = ctx

    def run(self, model, useLayout, dumpOutput=False):
        logging.info("Running TVM...")
        self.module.run()
        for i in range(len(model.outputs)):
            model.outputs[i].ndarray = self.module.get_output(
                i,
                tvm.nd.empty(model.outputs[i].shape,
                             model.outputs[i].dtype,
                             ctx=self.ctx)).asnumpy()
            out = self.module.get_output(
                i,
                tvm.nd.empty(model.outputs[i].shape,
                             model.outputs[i].dtype,
                             ctx=self.ctx)).asnumpy()
            model.outputs[i].ndarray = out
        if dumpOutput:
            model.store('output')

    def profile(self, target, times=50):
        logging.info("Evaluate inference time cost...")
        ftimer = self.module.module.time_evaluator('run',
                                                   self.ctx,
                                                   number=times)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        logging.info("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                     (np.mean(prof_res), np.std(prof_res)))

        # if not IsUpstream:
        #   if target.rpc.path:
        #     profile_path = target.rpc.path + 'profile.' + '.log'
        #     self.module.run_profile(False, 1, times, custom_filename=profile_path)
        #   else:
        #     self.module.run_profile(False, 1, times)


class Tuner:
    class Record:
        def __init__(self, dname, fname):
            self.best = "%s/%s.log" % (dname, fname)
            self.library = self.best + '.library'
            self.filter = self.best + '.filter'
            if not os.path.exists(dname):
                os.makedirs(dname)

        def pick_best(self, library=None):
            lib = self.library if library is None else library
            if os.path.exists(lib):
                autotvm.record.pick_best(lib, self.best)

            if os.path.exists(self.filter):
                with open(self.best, 'a') as out_file:
                    with open(self.filter, 'r') as in_file:
                        for line in in_file:
                            out_file.write(line)

            return os.path.exists(self.best)

    def __init__(self, name, target, n_trial=10, early_stopping=50):
        self.n_trial = n_trial
        self.early_stopping = early_stopping
        self.record = self.Record(target.rpc.key, name)
        if target.rpc.path:
            builder = autotvm.LocalBuilder(build_func='ndk', timeout=100)
            runner = autotvm.RPCRunner(target.rpc.key,
                                       host=target.tracker.ip,
                                       port=target.tracker.port,
                                       n_parallel=1,
                                       number=30,
                                       timeout=100)
        else:
            builder = autotvm.LocalBuilder(build_func='default', timeout=100)
            runner = autotvm.LocalRunner(number=30, timeout=100)
        self.measure_option = autotvm.measure_option(builder, runner)

    def tune(self, net, params, target):
        tuning_symbols = (relay.op.get('nn.conv2d'), relay.op.get('nn.dense'))
        tasks = autotvm.task.extract_from_program(net['main'],
                                                  ops=tuning_symbols,
                                                  params=params,
                                                  target_host=target.host,
                                                  target=target.target)
        # template_keys=template_key)
        tasks = self.filter_tasks(tasks)

        for i, tsk in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner_obj = XGBTuner(tsk, loss_type='rank')
            total = min(self.n_trial, len(tsk.config_space))
            tuner_obj.tune(n_trial=total,
                           early_stopping=self.early_stopping,
                           measure_option=self.measure_option,
                           callbacks=[
                               autotvm.callback.progress_bar(total,
                                                             prefix=prefix),
                               autotvm.callback.log_to_file(
                                   self.record.library)
                           ])

        self.record.pick_best()

    def filter_tasks(self, tasks):
        if os.path.exists(self.record.filter):
            filter = autotvm.apply_history_best(self.record.filter)
            filtered_tasks = []
            for task in tasks:
                if not filter._query_inside(task.target, task.workload):
                    filtered_tasks.append(task)
            return filtered_tasks
        else:
            return tasks


def get_model(name, dtype, useLayout, batch_size=1):
    """Get the symbol definition and random weight of a network"""
    builtin_workload = {
        'resnet-18': relay.testing.resnet.get_workload,
        'mobilenet': relay.testing.mobilenet.get_workload,
        'squeezenet': relay.testing.squeezenet.get_workload,
    }
    assert name in builtin_workload
    from .nn import Model, Tensor
    model = Model(name, dtype, useLayout)
    model.add('i', Tensor('data', (1, 224, 224, 3), dtype, layout=useLayout))
    model.add('o', Tensor('output', (1, 1000), dtype, layout=useLayout))
    net, params = builtin_workload[name](
        batch_size=batch_size,
        image_shape=model.inputs[0].shapeAs(useLayout)[1:],
        layout=useLayout)
    return net, params, model

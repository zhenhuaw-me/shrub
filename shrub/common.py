
class BaseRunner:
    def __init__(self, path: str):
        self.path = path
        self.model = None

    @property
    def quantized(self):
        raise NotImplementedError("Runner.quantized() not implemented!")

    def parse(self):
        raise NotImplementedError("Runner.parse() not implemented!")

    def run(self, inputs=None):
        raise NotImplementedError("Runner.run() not implemented!")

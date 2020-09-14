
class BaseRunner:
    DefaultLayout = None

    def __init__(self, path: str, layout=None):
        """ Runner for SOME model

        Parameters
        path: path to SOME model.
        layout: the input output layout of the model.
        """
        self.path = path
        self.layout = self.DefaultLayout if layout is None else layout
        self.model = None

    @property
    def quantized(self):
        raise NotImplementedError("Runner.quantized() not implemented!")

    def parse(self):
        raise NotImplementedError("Runner.parse() not implemented!")

    def run(self, inputs=None):
        raise NotImplementedError("Runner.run() not implemented!")

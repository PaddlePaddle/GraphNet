class GraphCompilerBackend:
    def __init__(self, config):
        self.config = config

    def __call__(self, model):
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()

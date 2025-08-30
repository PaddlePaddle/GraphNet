class GraphCompilerBackend:
    def __call__(self, model):
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()

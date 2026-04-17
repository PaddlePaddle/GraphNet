class GraphCompilerBackend:
    def __call__(self, model, input_spec=None):
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()

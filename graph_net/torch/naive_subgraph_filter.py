class GraphFilter:
    def __init__(self, config):
        self.config = config

    def __call__(self, gm, sample_inputs):
        return True

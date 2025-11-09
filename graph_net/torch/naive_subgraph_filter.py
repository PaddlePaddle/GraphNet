class GraphFilter:
    def __init__(self, config):
        self.config = config

    def __call__(self, gm, sample_inputs):
        print(f"GraphFilter\n{gm.code}")
        return True

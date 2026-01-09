class SampleRemoteExecutor:
    def __init__(self, machine: str, port: int):
        self.machine = machine
        self.port = port

    def __call__(self, model_path: str, random_seed: int) -> tuple:
        raise NotImplementedError("TODO")

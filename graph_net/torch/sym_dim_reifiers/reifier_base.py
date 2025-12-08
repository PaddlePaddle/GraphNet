class ReifierBase:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def get_reifier_name(self) -> bool:
        raise NotImplementedError()

    def match(self) -> bool:
        raise NotImplementedError()

    def reify(self):
        raise NotImplementedError()

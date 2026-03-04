import abc


class SamplePassMixin(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def declare_config(self):
        raise NotImplementedError()

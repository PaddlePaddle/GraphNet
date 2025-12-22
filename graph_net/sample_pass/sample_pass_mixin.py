import abc


class SamplePassMixin(abc.ABC):
    @abc.abstractmethod
    def declare_config(self):
        raise NotImplementedError()

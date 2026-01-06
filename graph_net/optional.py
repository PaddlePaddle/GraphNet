from typing import TypeVar, Generic, Union

T = TypeVar("T")


class Optional(Generic[T]):
    def __init__(self, value: Union[T, None]):
        self._value = value

    def reset(self, that):
        assert isinstance(that, Optional)
        self._value = that._value

    def is_some(self) -> bool:
        return self._value is not None

    def unwrap(self) -> T:
        """Returns the value or raises an error if None."""
        if self._value is None:
            raise ValueError("Tried to unwrap a None value!")
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Returns the value or a default if None."""
        return self._value if self._value is not None else default

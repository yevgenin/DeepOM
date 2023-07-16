from typing import TypeVar

import numpy as np

T = TypeVar('T')


class NDArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return np.asarray(v)

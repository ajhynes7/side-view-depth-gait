
from typing import Sequence, Tuple, TypeVar, Type
from numpy import ndarray


class BaseModel(object):

    def __init__(self):

        self.params = None


class LineModelND(BaseModel):

    def __init__(self):

        self.params: Tuple[ndarray, ndarray]


M = TypeVar('M', bound=BaseModel)


def ransac(data: Sequence, model_class: Type[M], min_samples: int, residual_threshold: float) -> Tuple[M, ndarray]: ...

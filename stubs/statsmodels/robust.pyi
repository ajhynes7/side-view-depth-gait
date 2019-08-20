
from typing import Callable, Union

from modules.typing import array_like


def mad(a: array_like, c: float = 0.675, axis: int = 0, center: Union[Callable, float] = None) -> float: ...

"""Custom types for type annotations."""

from typing import Any, Callable, Union, Sequence

import numpy as np


func_ab = Callable[[Any, Any], Any]

array_like = Union[Sequence, np.ndarray]

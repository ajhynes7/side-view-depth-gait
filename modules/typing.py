"""Custom types for type annotations."""

from typing import Any, Callable, Mapping, Union, Sequence

import numpy as np


adj_list = Mapping[int, Mapping[int, float]]

array_like = Union[Sequence, np.ndarray]

func_ab = Callable[[Any, Any], Any]

"""Custom types for type annotations."""

from typing import Any, Callable, Collection, Mapping, Union

import numpy as np


adj_list = Mapping[int, Mapping[int, float]]

array_like = Union[Collection, np.ndarray]

func_ab = Callable[[Any, Any], Any]

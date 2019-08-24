
from typing import Callable, Iterable, Iterator, Tuple

from modules.typing import array_like


class DataFrame:

    def __init__(self, data: Iterable): ...

    def assign(self, **kwargs) -> 'DataFrame': ...

    def empty(self) -> bool: ...

    def pipe(self, func: Callable, *args, **kwargs): ...

    def itertuples(self, index: bool = True, name: str = 'Pandas') -> Iterator: ...

    @property
    def shape(self) -> Tuple[int, ...]: ...


class Series:

    def __init__(self, array: array_like): ...


def notnull(x) -> bool: ...

def concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None,
        levels=None, names=None, verify_integrity=False, sort=None, copy=True): ...

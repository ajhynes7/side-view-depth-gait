
from typing import Sequence


class DataFrame:

    def __init__(self, array): ...

    def empty(self) -> bool: ...


class Series(Sequence):

    def __init__(self, array): ...


def notnull(x) -> bool: ...

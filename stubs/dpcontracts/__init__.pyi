
from typing import Callable


def require(msg: str, func: Callable) -> Callable: ...

def ensure(msg: str, func: Callable) -> Callable: ...

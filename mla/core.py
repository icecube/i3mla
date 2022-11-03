"""Docstring"""

from .configurable import Configurable


def generate_default_config(classes: list) -> dict:
    """Docstring"""
    to_return = {}
    for c in classes:
        if type(c) is tuple:
            cls = c[0]
            name = c[1]
        else:
            cls = c
            name = str(cls.__name__)
        if not issubclass(cls, Configurable):
            continue
        to_return[name] = cls.default_config()
    return to_return

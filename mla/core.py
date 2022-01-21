"""Docstring"""

from .configurable import Configurable

def generate_default_config(classes: list) -> dict:
    """Docstring"""
    return {
        c.__name__: c.generate_config() for c in classes if issubclass(c, Configurable)}

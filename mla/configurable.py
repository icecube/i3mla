"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar
from abc import ABCMeta, abstractmethod
import inspect
import dataclasses

@dataclasses.dataclass(kw_only=True)
class Configurable(metaclass=ABCMeta):
    """Docstring"""
    @classmethod
    def default_config(cls) -> dict:
        """Docstring"""
        sig = inspect.signature(cls.__init__)
        defaults = {
            param.name: param.default
            for param in sig.parameters.values()
            if (param.default is not param.empty) and (type(param.default) is not property)
        }
        to_change = {}
        for var, default in defaults.items():
            if type(default) == dataclasses._HAS_DEFAULT_FACTORY_CLASS:
                to_change[var] = next(f for f in dataclasses.fields(cls)
                     if f.name == var).default_factory()
        for var, default in to_change.items():
            defaults[var] = default
        return {'class': str(cls.__name__), 'args': defaults}

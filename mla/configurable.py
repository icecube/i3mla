"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from abc import ABCMeta, abstractmethod
import inspect

class Configurable(metaclass=ABCMeta):
    """Docstring"""
    _config_map = {}

    @classmethod
    def default_config(cls) -> dict:
        """Docstring"""
        sig = inspect.signature(cls.__init__)
        defaults = {
            param: param.default
            for param in sig.parameters.values()
            if param.default is not param.empty
        }
        return {key: defaults[var] for var, key in cls._config_map.items()}

    @property
    def config(self) -> dict:
        """Docstring"""
        return {
            key: getattr(self, var)
            for var, key in self.__class__._config_map.items()
        }

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict, *args) -> 'Configurable':
        """Docstring"""

    @classmethod
    def _map_kwargs(cls, config: dict) -> dict:
        """Docstring"""
        return {var: config[key] for var, (key, _) in cls._config_map.items()}

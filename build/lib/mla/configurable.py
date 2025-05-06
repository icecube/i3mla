"""Docstring"""

__author__ = 'John Evans and Jason Fan'
__copyright__ = 'Copyright 2024'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '1.4.1'
__maintainer__ = 'Jason Fan'
__email__ = 'klfan@terpmail.umd.edu'
__status__ = 'Development'

import dataclasses


@dataclasses.dataclass
class Configurable:
    """Docstring"""
    config: dict

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {}

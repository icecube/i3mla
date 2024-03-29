"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
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

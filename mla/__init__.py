"""__init__.py"""
# flake8: noqa
from .analysis import *
from .configurable import *
from .data_handlers import *
from .minimizers import *
from .params import *
from .sob_terms import *
from .sources import *
from .test_statistics import *
from .time_profiles import *
from .trial_generators import *
from .utility_functions import *

def generate_default_config(classes: list) -> dict:
    """Docstring"""
    return {
        c.__name__: c.generate_config() for c in classes if issubclass(c, Configurable)}

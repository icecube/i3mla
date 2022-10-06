"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import ClassVar, List, Tuple
from dataclasses import InitVar, dataclass, field

from .configurable import Configurable
from .data_handlers import DataHandler, Injector
from .minimizers import MinimizerFactory
from .params import Params
from .sources import PointSource
from .test_statistics import LLHTestStatisticFactory
from .trial_generators import SingleSourceTrialGenerator


@dataclass(kw_only=True)
class SingleSourceLLHAnalysis:
    """Docstring"""
    minimizer_factory: MinimizerFactory
    test_statistic_factory: LLHTestStatisticFactory
    injector_source: InitVar[Tuple[Injector, PointSource]]
    trial_generator: SingleSourceTrialGenerator
    _injector_source: Tuple[DataHandler, PointSource] = field(init=False, repr=False)

    def __post_init__(self, injector_source: Tuple[Injector, PointSource]) -> None:
        self._injector_source = injector_source

    def produce_and_minimize(
        self,
        params: Params,
        fitting_params: List[str],
        n_signal: float = 0,
    ) -> dict:
        """Docstring"""
        trial = self.trial_generator(n_signal=n_signal)
        test_statistic = self.test_statistic_factory(params, trial)
        minimizer = self.minimizer_factory(test_statistic)
        return minimizer(fitting_params)

    def generate_params(self) -> Params:
        """Docstring"""
        return self.test_statistic_factory.generate_params()

    @property
    def injector_source(self) -> Tuple[Injector, PointSource]:
        """Docstring"""
        return self._injector_source

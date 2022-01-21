"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'
from typing import List, Tuple, Type
from dataclasses import dataclass, field

from .core import generate_default_config
from .data_handlers import DataHandler
from .minimizers import Minimizer
from .params import Params
from .sob_terms import SoBTermFactory
from .sources import PointSource
from .test_statistics import LLHTestStatisticFactory
from .trial_generators import SingleSourceTrialGenerator


@dataclass
class SingleSourceLLHAnalysis:
    """Docstring"""
    config: dict
    minimizer_class: Type[Minimizer]
    sob_term_factories: List[SoBTermFactory]
    data_handler_source: Tuple[DataHandler, PointSource]
    _sob_term_factories: List[SoBTermFactory] = field(init=False, repr=False)
    _data_handler_source: Tuple[DataHandler, PointSource] = field(init=False, repr=False)
    _trial_generator: SingleSourceTrialGenerator = field(init=False, repr=False)
    _test_statistic_factory: LLHTestStatisticFactory = field(init=False, repr=False)

    def produce_and_minimize(
        self,
        params: Params,
        fitting_params: List[str],
        n_signal: float = 0,
    ) -> tuple:
        """Docstring"""
        trial = self._trial_generator(n_signal=n_signal)
        test_statistic = self._test_statistic_factory(params, trial)
        minimizer = self.minimizer_class(
            self.config[self.minimizer_class.__name__], test_statistic)
        return minimizer(fitting_params)

    @property
    def sob_term_factories(self) -> List[SoBTermFactory]:
        """Docstring"""
        return self._sob_term_factories

    @sob_term_factories.setter
    def sob_term_factories(self, sob_term_factories: List[SoBTermFactory]) -> None:
        """Docstring"""
        self._sob_term_factories = sob_term_factories
        self._test_statistic_factory = LLHTestStatisticFactory(  # pylint: disable=too-many-function-args
            self.config['LLHTestStatisticFactory'],
            self._sob_term_factories,
        )

    @property
    def data_handler_source(self) -> Tuple[DataHandler, PointSource]:
        """Docstring"""
        return self._data_handler_source

    @data_handler_source.setter
    def data_handler_source(
            self, data_handler_source: Tuple[DataHandler, PointSource]) -> None:
        """Docstring"""
        self._data_handler_source = data_handler_source
        self._trial_generator = SingleSourceTrialGenerator(
            self.config['SingleSourceTrialGenerator'],
            *self._data_handler_source,
        )

    @classmethod
    def generate_default_config(cls, minimizer_class: Type[Minimizer]) -> dict:
        """Docstring"""
        return generate_default_config([
            minimizer_class,
            SingleSourceTrialGenerator,
            LLHTestStatisticFactory,
        ])

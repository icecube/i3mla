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
from dataclasses import dataclass, field

from .configurable import Configurable
from .data_handlers import DataHandler
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
    data_handler_source: Tuple[DataHandler, PointSource]
    trial_generator: SingleSourceTrialGenerator
    _data_handler_source: Tuple[DataHandler, PointSource] = field(init=False, repr=False)

    def produce_and_minimize(
        self,
        params: Params,
        fitting_params: List[str],
        n_signal: float = 0,
    ) -> tuple:
        """Docstring"""
        trial = self.trial_generator(n_signal=n_signal)
        test_statistic = self.test_statistic_factory(params, trial)
        minimizer = self.minimizer_factory(test_statistic)
        return minimizer(fitting_params)

    def generate_params(self) -> Params:
        """Docstring"""
        return self.test_statistic_factory.generate_params()

    @property
    def data_handler_source(self) -> Tuple[DataHandler, PointSource]:
        """Docstring"""
        return self._data_handler_source

    @data_handler_source.setter
    def data_handler_source(
            self, data_handler_source: Tuple[DataHandler, PointSource]) -> None:
        """Docstring"""
        self._data_handler_source = data_handler_source
        self.trial_generator.data_handler_source = data_handler_source


@dataclass(kw_only=True)
class SingleSourceFlareStackLLHAnalysis(SingleSourceLLHAnalysis, Configurable):
    """Docstring"""
    _config_map: ClassVar[dict] = {
        '_min_sob': ('Minimum Signal-over-background Ratio For Flare', 1),
        '_min_length': ('Minimum Flare Duration (days)', 1),
    }

    _min_sob: float = 1
    _min_length: float = 1

    @classmethod
    def from_config(
        cls,
        config: dict,
        minimizer_factory: MinimizerFactory,
        test_statistic_factory: LLHTestStatisticFactory,
        data_handler_source: Tuple[DataHandler, PointSource],
        trial_generator: SingleSourceTrialGenerator,
    ) -> 'SingleSourceFlareStackLLHAnalysis':
        kwargs = {var: config[key] for var, (key, _) in cls._config_map.items()}
        return cls(
            minimizer_factory=minimizer_factory,
            test_statistic_factory=test_statistic_factory,
            data_handler_source=data_handler_source,
            trial_generator=trial_generator,
            **cls._map_kwargs(config),
        )

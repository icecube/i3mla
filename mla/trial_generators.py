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
from typing import ClassVar, Optional, Tuple

import numpy as np
import numpy.lib.recfunctions as rf

from . import utility_functions as uf
from .configurable import Configurable
from .data_handlers import DataHandler
from .sources import PointSource


@dataclasses.dataclass(kw_only=True)
class SingleSourceTrialGenerator(Configurable):
    """Docstring"""
    data_handler: DataHandler
    source: PointSource

    _config_map: ClassVar[dict] = {
        '_random_seed': ('Random Seed', None),
        '_fixed_ns': ('Fixed n_s', False),
    }

    _random_seed: Optional[int] = None 
    _fixed_ns: bool = False 

    _data_handler: DataHandler = dataclasses.field(init=False, repr=False)
    _source: PointSource = dataclasses.field(init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: dict,
        data_handler: DataHandler,
        source: PointSource,
    ) -> 'SingleSourceTrialGenerator':
        """Docstring"""
        return cls(
            data_handler=data_handler,
            source=source,
            **cls._map_kwargs(config),
        )

    def __call__(self, n_signal: float = 0) -> np.ndarray:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            n_signal: flux norm if not fixed_ns

        Returns:
            An array of combined signal and background events.
        """
        rng = np.random.default_rng(self._random_seed)
        n_background = rng.poisson(self.data_handler.n_background)
        if not self._fixed_ns:
            n_signal = rng.poisson(self.data_handler.calculate_n_signal(n_signal))

        background = self.data_handler.sample_background(n_background, rng)
        background['ra'] = rng.uniform(0, 2 * np.pi, len(background))

        if n_signal > 0:
            signal = self.data_handler.sample_signal(int(n_signal), rng)
            signal = self._rotate_signal(signal)
        else:
            signal = np.empty(0, dtype=background.dtype)

        # Because we want to return the entire event and not just the
        # number of events, we need to do some numpy magic. Specifically,
        # we need to remove the fields in the simulated events that are
        # not present in the data events. These include the true direction,
        # energy, and 'oneweight'.
        signal = rf.drop_fields(
            signal, [n for n in signal.dtype.names if n not in background.dtype.names])

        # Combine the signal background events and time-sort them.
        # Use recfunctions.stack_arrays to prevent numpy from scrambling entry order
        return rf.stack_arrays([background, signal], autoconvert=True, usemask=False)

    def _rotate_signal(self, signal: np.ndarray) -> np.ndarray:
        """Docstring"""
        ra, dec = self.source.sample(len(signal))

        signal['ra'], signal['dec'] = uf.rotate(
            signal['trueRa'],
            signal['trueDec'],
            ra,
            dec,
            signal['ra'],
            signal['dec'],
        )

        signal['trueRa'], signal['trueDec'] = uf.rotate(
            signal['trueRa'],
            signal['trueDec'],
            ra,
            dec,
            signal['trueRa'],
            signal['trueDec'],
        )

        signal['sindec'] = np.sin(signal['dec'])
        return signal

    @property
    def source(self) -> PointSource:
        """Docstring"""
        return self._source

    @source.setter
    def source(self, source: PointSource) -> None:
        """Docstring"""
        self._data_handler.dec_cut_loc = source.location[1]
        self._source = source

    @property
    def data_handler(self) -> DataHandler:
        """Docstring"""
        return self._data_handler

    @data_handler.setter
    def data_handler(self, data_handler: DataHandler) -> None:
        """Docstring"""
        self._data_handler = data_handler
        self._data_handler.dec_cut_loc = self._source.location[1]

    @property
    def data_handler_source(self) -> Tuple[DataHandler, PointSource]:
        """Docstring"""
        return (self._data_handler, self._source)

    @data_handler_source.setter
    def data_handler_source(
            self, data_handler_source: Tuple[DataHandler, PointSource]) -> None:
        """Docstring"""
        self._data_handler = data_handler_source[0]
        self._source = data_handler_source[1]
        self._data_handler.dec_cut_loc = self._source.location[1]

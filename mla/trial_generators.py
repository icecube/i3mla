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

from . import utility_functions as uf
from .configurable import Configurable
from .data_handlers import Injector
from .sources import PointSource
from .events import Events, SimEvents


@dataclasses.dataclass(kw_only=True)
class SingleSourceTrialGenerator(Configurable):
    """
        random_seed: Random Seed
        fixed_ns: Fixed n_s
    """
    injector: Injector
    source: PointSource

    random_seed: Optional[int] = None
    fixed_ns: bool = False

    def __call__(self, n_signal: float = 0) -> Events:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            n_signal: flux norm if not fixed_ns

        Returns:
            An array of combined signal and background events.
        """
        rng = np.random.default_rng(self.random_seed)
        n_background = rng.poisson(self.injector.n_background)
        if not self.fixed_ns:
            n_signal = rng.poisson(self.injector.calculate_n_signal(n_signal))

        background = self.injector.sample_background(n_background, rng)
        background.ra = rng.uniform(0, 2 * np.pi, len(background))

        if n_signal > 0:
            signal = self.injector.sample_signal(int(n_signal), rng)
            signal = self._rotate_signal(signal)
        else:
            signal = SimEvents.empty()

        # Because we want to return the entire event and not just the
        # number of events, we need to do some numpy magic. Specifically,
        # we need to remove the fields in the simulated events that are
        # not present in the data events. These include the true direction,
        # energy, and 'oneweight'.
        signal = signal.to_events()

        # Combine the signal background events and time-sort them.
        # Use recfunctions.stack_arrays to prevent numpy from scrambling entry order
        events = Events.concatenate([signal, background])
        events.sort('time')
        return events

    def _rotate_signal(self, signal: SimEvents) -> SimEvents:
        """Docstring"""
        ra, dec = self.source.sample(len(signal))

        signal.ra, signal.dec = uf.rotate(
            signal.trueRa,
            signal.trueDec,
            ra,
            dec,
            signal.ra,
            signal.dec,
        )

        signal.trueRa, signal.trueDec = uf.rotate(
            signal.trueRa,
            signal.trueDec,
            ra,
            dec,
            signal.trueRa,
            signal.trueDec,
        )

        return signal

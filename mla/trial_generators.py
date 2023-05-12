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
from typing import Optional

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
    rng: Optional[np.random.Generator] = None
    fixed_ns: bool = False

    def __post_init__(self) -> None:
        """Docstring"""
        if self.rng is not None:
            return
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, n_signal: float = 0) -> Events:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            n_signal: flux norm if not fixed_ns

        Returns:
            An array of combined signal and background events.
        """
        background = self.generate_background_unsorted()
        signal = self.generate_signal_unsorted(n_signal)

        # Combine the signal background events and time-sort them.
        # Use recfunctions.stack_arrays to prevent numpy from scrambling entry order
        events = Events.concatenate([signal, background])
        events.sort('time')
        return events

    def generate_background_unsorted(self) -> Events:
        """Docstring

        CAUTION: Events are assumed to bee sorted by time. This function does not do that
        sorting for you.
        """
        n_background = self.rng.poisson(self.injector.n_background)
        background = self.injector.sample_background(n_background, self.rng)
        background.ra = self.rng.uniform(0, 2 * np.pi, len(background))
        return background

    def generate_signal_unsorted(self, n_signal: float = 0) -> Events:
        """Docstring

        CAUTION: Events are assumed to bee sorted by time. This function does not do that
        sorting for you.
        """
        if not self.fixed_ns:
            n_signal = self.rng.poisson(self.injector.calculate_n_signal(n_signal))

        if n_signal > 0:
            signal = self.injector.sample_signal(int(n_signal), self.rng)
            signal = self._rotate_signal(signal)
        else:
            signal = SimEvents.empty()

        # Because we want to return the entire event and not just the
        # number of events, we need to do some numpy magic. Specifically,
        # we need to remove the fields in the simulated events that are
        # not present in the data events. These include the true direction,
        # energy, and 'oneweight'.
        signal = signal.to_events()
        return signal

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


@dataclasses.dataclass(kw_only=True)
class StackedTrialGenerator(Configurable):
    """Docstring"""
    trial_generator_dict: dict[str, SingleSourceTrialGenerator]
    random_seed: Optional[int] = None
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        """Docstring"""
        if self.rng is not None:
            return
        self.rng = np.random.default_rng(self.random_seed)

        for _, tg in self.trial_generator_dict.items():
            tg.rng = self.rng

    def __call__(self, n_signal_dict : dict[str, float] = 0):
        """Docstring"""
        # only need one background generator
        background = list(
                self.trial_generator_dict.values())[0].generate_background_unsorted()

        signals = [
            tg.generate_signal_unsorted(n_signal_dict[key])
            for key, tg in self.trial_generator_dict.items()
        ]

        events = Events.concatenate([background, *signals])
        events.sort('time')
        return events

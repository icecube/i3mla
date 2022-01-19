"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import TYPE_CHECKING

import dataclasses

import numpy as np
import numpy.lib.recfunctions as rf

from . import utility_functions as uf

if TYPE_CHECKING:
    from .data_handlers import DataHandler
    from .sources import Source
else:
    DataHandler = object
    Source = object

@dataclasses.dataclass
class SingleSourceTrialGenerator:
    """Docstring"""
    config: dict
    data_handler: DataHandler
    source: Source

    def __call__(self, n_signal: float = 0) -> np.ndarray:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            n_signal: flux norm if not fixed_ns

        Returns:
            An array of combined signal and background events.
        """
        n_background = np.random.poisson(self.data_handler.n_background)
        if not self.config['fixed_ns']:
            n_signal = np.random.poisson(self.data_handler.calculate_n_signal(n_signal))

        if self.config['random_seed'] is not None:
            np.random.seed(self.config['random_seed'])

        background = self.data_handler.sample_background(n_background)
        background['ra'] = np.random.uniform(0, 2 * np.pi, len(background))

        if n_signal > 0:
            signal = self.data_handler.sample_signal(n_signal)
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
        ra, dec = self.source.sample_location(len(signal))

        signal['ra'], signal['dec'] = uf.rotate(
            signal['truera'],
            signal['truedec'],
            ra,
            dec,
            signal['ra'],
            signal['dec'],
        )

        signal['truera'], signal['truedec'] = uf.rotate(
            signal['truera'],
            signal['truedec'],
            ra,
            dec,
            signal['truera'],
            signal['truedec'],
        )

        signal['sindec'] = np.sin(signal['dec'])
        return signal

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {
            'random_seed': None,
            'fixed_ns': False,
        }

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

import numpy as np
import numpy.lib.recfunctions as rf

from . import utility_functions as uf
from . import configurable

from .data_handlers import DataHandler
from .sources import PointSource


@dataclasses.dataclass
class SingleSourceTrialGenerator(configurable.Configurable):
    """Docstring"""

    data_handler: DataHandler
    source: PointSource
    _source: PointSource = dataclasses.field(init=False, repr=False)

    def __call__(self, n_signal: float = 0) -> np.ndarray:
        """Produces a single trial of background+signal events based on inputs.

        Args:
            n_signal: flux norm if not fixed_ns

        Returns:
            An array of combined signal and background events.
        """
        rng = np.random.default_rng(self.config["random_seed"])
        n_background = rng.poisson(self.data_handler.n_background)
        if not self.config["fixed_ns"]:
            n_signal = rng.poisson(self.data_handler.calculate_n_signal(n_signal))

        background = self.data_handler.sample_background(n_background, rng)
        background["ra"] = rng.uniform(0, 2 * np.pi, len(background))

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
            signal, [n for n in signal.dtype.names if n not in background.dtype.names]
        )

        # Combine the signal background events and time-sort them.
        # Use recfunctions.stack_arrays to prevent numpy from scrambling entry order
        if background.dtype == signal.dtype:
            return np.concatenate([background, signal])
        else:
            return rf.stack_arrays(
                [background, signal], autoconvert=True, usemask=False
            )

    def _rotate_signal(self, signal: np.ndarray) -> np.ndarray:
        """Docstring"""
        ra, dec = self.source.sample(len(signal))

        signal["ra"], signal["dec"] = uf.rotate(
            signal["trueRa"],
            signal["trueDec"],
            ra,
            dec,
            signal["ra"],
            signal["dec"],
        )

        signal["trueRa"], signal["trueDec"] = uf.rotate(
            signal["trueRa"],
            signal["trueDec"],
            ra,
            dec,
            signal["trueRa"],
            signal["trueDec"],
        )

        signal["sindec"] = np.sin(signal["dec"])
        return signal

    @property
    def source(self) -> PointSource:
        """Docstring"""
        return self._source

    @source.setter
    def source(self, source: PointSource) -> None:
        """Docstring"""
        self.data_handler.dec_cut_location = source.config["dec"]
        self._source = source

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config["random_seed"] = None
        config["fixed_ns"] = False
        return config

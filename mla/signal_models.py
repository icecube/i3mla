"""
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2021 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import Optional, Tuple, Union

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar

import scipy.stats
import numpy as np
import numpy.lib.recfunctions as rf

from . import sources
from . import time_profiles
from . import utility_functions as uf


@dataclass
class BaseSignalModel:
    """Stores the events and pre-processed parameters used in analyses.

    Currently, this class uses internal data and monte-carlo datasets. This
    will be updated before the first release to use the upcoming public data
    release.

    Attributes:
        sim (np.ndarray): Simulated neutrino events.
        gamma (float):
    """
    sim: np.ndarray
    source: sources.PointSource
    config: dict

    def __post_init__(self, sim) -> None:
        """Docstring"""
        if 'sindec' not in self.sim.dtype.names:
            self.sim = rf.append_fields(
                self.sim,
                'sindec',
                np.sin(self.sim['dec']),
                usemask=False,
            )

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        return {}


@dataclass
class PowerLawSignalModel(BaseSignalModel):
    """Docstring"""
    def __post_init__(self) -> None:
        """Docstring"""
        if 'weight' not in self.sim.dtype.names:
            self.sim = rf.append_fields(
                self.sim, 'weight',
                np.zeros(len(self.sim)),
                dtypes=np.float32
            )

        # assign the weights using the newly defined "time profile"
        # classes above. if you want to make this a more complicated
        # shape, talk to me and we can work it out.
        rescaled_energy = (
            self.sim['trueE'] / self.config['normalization_energy (GeV)']
        )**self.config['gamma']
        self.sim['weight'] = self.sim['ow'] * rescaled_energy

    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['gamma'] = -2
        config['normalization_energy (GeV)'] = 100.e3
        return config

    def get_ns(self, time_integrated_flux: float) -> float:
        """Docstring"""
        ns = self.sim['weight'].sum() * time_integrated_flux
        return ns

    def inject_signal_events(
        self,
        flux_norm: float,
        n_signal_observed: Optional[int] = None,
    ) -> np.ndarray:
        """injects signal events for a trial.

        args:
            flux_norm:
            n_signal_observed:

        returns:
            an array of injected signal events.
        """

        # pick the signal events
        total = self.sim['weight'].sum()

        if n_signal_observed is None:
            n_signal_observed = scipy.stats.poisson.rvs(total * flux_norm)

        signal = np.random.choice(
            self.sim,
            n_signal_observed,
            p=self.sim['weight'] / total,
            replace=False,
        ).copy()

        if len(signal) > 0:
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


@dataclass
class TimeDependentMixin(BaseSignalModel):
    """Docstring"""
    @classmethod
    def generate_config(cls):
        """Docstring"""
        config = super().generate_config()
        config['time_profile'] = None
        return config


@dataclass
class CutSimByDecMixin(BaseSignalModel):
    """stores the events and pre-processed parameters used in analyses.

    currently, this class uses internal data and monte-carlo datasets. this
    will be updated before the first release to use the upcoming public data
    release.
    """
    def __post_init__(self) -> None:
        """Docstring"""
        super().__post_init__()
        sindec_dist = np.abs(self.source.dec - self.sim['truedec'])
        close = sindec_dist < self.config['dec_band_width']
        self.sim = self.sim[close]

        omega = 2 * np.pi * (np.min(
            [np.sin(self.source.dec + self.config['dec_band_width']), 1]
        ) - np.max([np.sin(self.source.dec - self.config['dec_band_width']), -1]))
        self.sim['ow'] /= omega


    @classmethod
    def generate_config(cls) -> dict:
        """Docstring"""
        config = super().generate_config()
        config['dec_band_width'] = np.deg2rad(3)
        return config

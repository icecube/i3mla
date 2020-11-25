__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

"""
Docstring
"""

from typing import Dict, Optional

import numpy as np
import scipy
import scipy.stats

from mla import tools
from mla import models
from mla import spectral
import numpy.lib.recfunctions as rf

class PsInjector:
    """Class info...
    
    More class info...
    
    Attributes:
        source (Dict[str, float]):
        event_model (EventModel):
    """
    def __init__(self, source: Dict[str, float]) -> None:
        """Function info...
        
        More function info...
        """
        self.source = source
        
    def signal_spatial_pdf(self, events: np.ndarray) -> np.array:
        """Calculates the signal probability of events based on their angular distance from a source.
        
        More function info...
        
        Args:
            events: An array of events including their positional data.
            
        Returns:
            The value for the signal space pdf for the given events angular distances.
        """
        sigma = events['angErr']
        x = tools.angular_distance(events['ra'], events['dec'], self.source['ra'], self.source['dec'])
        return (1.0/(2*np.pi*sigma**2))*np.exp(-x**2/(2*sigma**2))
    
    def background_spatial_pdf(self, events: np.ndarray, event_model: models.EventModel) -> np.array:
        """Calculates the background probability of events based on their declination.
        
        More function info...
        
        Args:
            events: An array of events including their declination.
            
        Returns:
            The value for the background space pdf for the given events declinations.
        """
        return (1/(2*np.pi))*event_model.background_dec_spline(np.sin(events['dec']))
    
    def reduced_sim(self,
                    livetime: float,
                    event_model: Optional[models.EventModel] = None,
                    sim: Optional[np.ndarray] = None,
                    flux_norm: Optional[float] = 0, 
                    gamma: Optional[float] = -2,
                    spectrum: Optional[spectral.BaseSpectrum] = None,
                    sampling_width: Optional[float] = None) -> np.ndarray:
        """Function info...
        
        Prunes the simulation set to only events close to a given source and calculate the
        weight for each event. Adds the weights as a new column to the simulation set.
            
        Args:
            livetime: livetime in days
            event_model: models.EventModel object that holds the simulation set
            sim: simulation set. Alternative for user not passing event_model. 
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            spectrum: spectral.BaseSpectrum object.Alternative for user not passing flux_norm and gamma. 
            sampling_width: The bandwidth around the source declination to cut events.
        
        Returns:
            A reweighted simulation set around the source declination.
        """
        # Pick out only those events that are close in
        # declination. We only want to sample from those.
        if event_model is not None:
            sim = event_model.sim
        
        if sampling_width is not None:
            sindec_dist = np.abs(self.source['dec']-sim['trueDec'])
            close = sindec_dist < sampling_width
            
            reduced_sim = rf.append_fields(
                sim[close].copy(),
                'weight',
                np.zeros(close.sum()),
                dtypes=np.float32,
                usemask=False)
            omega = 2*np.pi * (np.min([np.sin(self.source['dec']+sampling_width), 1]) -\
                    np.max([np.sin(self.source['dec']-sampling_width), -1]))
        else:
            reduced_sim = rf.append_fields(
                sim.copy(),
                'weight',
                np.zeros(len(sim)),
                dtypes=np.float32,
                usemask=False)
            omega = 4*np.pi

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        if spectrum is not None:
            reduced_sim['weight'] = reduced_sim['ow'] * spectrum(reduced_sim['trueE']) * livetime * 3600 * 24
        else:
            reduced_sim['weight'] = reduced_sim['ow'] * flux_norm * (reduced_sim['trueE']/100.e3)**gamma * 3600 * 24

        # Apply the sampling width, which ensures that we
        # sample events from similar declinations.
        # When we do this, correct for the solid angle
        # we're including for sampling
        reduced_sim['weight'] /= omega
        return reduced_sim
    
    def inject_background_events(self, event_model: models.EventModel) -> np.ndarray:
        """Function info...
        
        More function info...
        
        Returns:
            An array of injected background events.
        """
        # Get the number of events we see from these runs
        n_background = event_model.grl['events'].sum()
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(event_model.data, n_background_observed).copy()

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2*np.pi, len(background))
        
        return background
    
    def inject_signal_events(self, reduced_sim: np.ndarray) -> np.ndarray:
        """Function info...
        
        More function info...
        
        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            
        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        total = reduced_sim['weight'].sum()

        n_signal_observed = scipy.stats.poisson.rvs(total)
        signal = np.random.choice(
            reduced_sim, 
            n_signal_observed,
            p=reduced_sim['weight']/total,
            replace = False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones*self.source['ra'], ones*self.source['dec'],
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones*self.source['ra'], ones*self.source['dec'],
                signal['trueRa'], signal['trueDec'])
        
        return signal
    
    def inject_nsignal_events(self, reduced_sim: np.ndarray, n: int) -> np.ndarray:
        """Function info...
        
        Inject n signal events.
        
        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            n: Number of events.
            
        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        total = reduced_sim['weight'].sum()

        n_signal_observed = n
        signal = np.random.choice(
            reduced_sim, 
            n_signal_observed,
            p=reduced_sim['weight']/total,
            replace = False).copy()

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones*self.source['ra'], ones*self.source['dec'],
                signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(
                signal['trueRa'], signal['trueDec'],
                ones*self.source['ra'], ones*self.source['dec'],
                signal['trueRa'], signal['trueDec'])
        
        return signal
    
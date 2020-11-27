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
from mla import time_profiles
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
        self.signal_time_profile = None
        self.background_time_profile = None
        
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
    
    def inject_background_events_No_time(self, event_model: models.EventModel) -> np.ndarray:
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

    def inject_background_events(self,
                                 event_model: models.EventModel,
                                 background_time_profile: Optional[time_profiles.GenericProfile] = None,
                                 background_window: Optional[float] = 14
                                 ) -> np.ndarray:
        """Function info...
        
        Inject Background Events
        
        Args:
            event_model: EventModel that holds the grl and data
            background_time_profile: Background time profile to do the injection
            background_window: Days used to estimated the background rate
        
        Returns:
            An array of injected background events.
        """
        if background_time_profile is not None:
            self.set_background_profile(event_model, background_time_profile, background_window)
            
        # Get the number of events we see from these runs
        n_background = self.n_background
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(event_model.data, n_background_observed).copy()

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2*np.pi, len(background))
        
        #Randomize the background time
        if self.background_time_profile is not None:
            background['time'] = self.background_time_profile.random(size = len(background))
        return background
    
    def set_background_profile(self, 
                       event_model: models.EventModel, 
                       background_time_profile: time_profiles.GenericProfile,
                       background_window: Optional[float] = 14) -> None:
        """Function info...
        
        Setting the background rate for the models
        Args:
            event_model: EventModel that holds the grl
            background_window: Days used to estimated the background rate
            background_time_profile: Background time profile to do the injection
        
        Returns:
            None
        """
        # Find the run contian in the background time window
        start_time = background_time_profile.get_range()[0]
        fully_contained = (event_model.grl['start'] >= start_time-background_window) &\
                            (event_model.grl['stop'] < start_time)
        start_contained = (event_model.grl['start'] < start_time-background_window) &\
                            (event_model.grl['stop'] > start_time-background_window)
        background_runs = (fully_contained | start_contained)
        if not np.any(background_runs):
            print("ERROR: No runs found in GRL for calculation of "
                  "background rates!")
            raise RuntimeError
        background_grl = event_model.grl[background_runs]
        
        #Find the background rate
        n_background = background_grl['events'].sum()
        n_background /= background_grl['livetime'].sum()
        n_background *= background_time_profile.effective_exposure()
        self.n_background = n_background
        self.background_time_profile = background_time_profile
        return
    
    def set_signal_profile(self, 
                       signal_time_profile: time_profiles.GenericProfile) -> None:
        """Function info...
        
        Setting the signal time profile
        Args:
            signal_time_profile: Signal time profile to do the injection
        
        Returns:
            None
        """
        self.signal_time_profile = signal_time_profile
        return
    
    
    def inject_signal_events(self, 
                             reduced_sim: np.ndarray, 
                             signal_time_profile:Optional[time_profiles.GenericProfile] = None) -> np.ndarray:
        """Function info...
        
        More function info...
        
        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            signal_time_profile: The time profile of the injected signal
            
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
        #set the signal time profile if provided
        if signal_time_profile is not None:
            self.set_signal_profile(signal_time_profile)
         
        #Randomize the signal time
        if self.signal_time_profile is not None:    
            signal['time'] = self.signal_time_profile.random(size = len(signal))
        
        return signal
    
    def inject_nsignal_events(self, 
                              reduced_sim: np.ndarray, 
                              n: int,
                              signal_time_profile:Optional[time_profiles.GenericProfile] = None) -> np.ndarray:
        """Function info...
        
        Inject n signal events.
        
        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            n: Number of events.
            signal_time_profile: The time profile of the injected signal
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
                
        #set the signal time profile if provided
        if signal_time_profile is not None:
            self.set_signal_profile(signal_time_profile)
         
        #Randomize the signal time
        if self.signal_time_profile is not None:    
            signal['time'] = self.signal_time_profile.random(size = len(signal))
        
        return signal
    
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

import scipy
import numpy as np
from tqdm import tqdm
from i3pubtools import tools
from i3pubtools import time_profiles

import numpy.lib.recfunctions as rf
import scipy.interpolate

class PsFlareLLH:
    """Performs an point-source analysis assuming some single-flaring behavior to the signal.
    
    More class info...
    
    Attributes:
        data (np.ndarray): Real neutrino event data.
        sim (np.ndarray): Simulated neutrino events.
        grl (np.ndarray): A list of runs/times when the detector was working properly.
        bins (np.ndarray): Bin edges for dec and logE for generating sob maps.
        gammas (np.array): Gamma steps for generating sob maps.
        signal_time_profile (generic_profile.GenericProfile): A time profile for the background distribution.
        background_time_profile (generic_profile.GenericProfile): A time profile for the signal distribution.
        source (dict): Where to look for neutrinos.
    """

    def __init__(self, 
                 data: np.ndarray, 
                 sim: np.ndarray, 
                 grl: np.ndarray, 
                 gammas: np.ndarray, 
                 bins: np.ndarray, 
                 signal_time_profile: Optional[time_profiles.GenericProfile] = None,
                 background_time_profile: Optional[time_profiles.GenericProfile] = None,
                 source: Dict[float, float] = {'ra':np.pi/2, 'dec':np.pi/6},
                 infile: Optional[str] = None, 
                 outfile: Optional[str] = None
    ) -> None:
        """Inits PsFlareLLH and calculates sob maps.
        
        More function info...

        Args:
            infile: A numpy file of precomputed maps from another PSFlareLLH object.
            outfile: Where to save the computed maps.
        """
        
        if signal_time_profile is None:
            signal_time_profile = time_profiles.GaussProfile(np.average(data['time']), np.std(data['time'])),
        if background_time_profile is None:
            background_time_profile = time_profiles.UniformProfile(np.min(data['time']), np.max(data['time'])),
        
        self.data = data
        self.sim = sim
        self.grl = grl
        self.bins = bins
        self.gammas = gammas
        self.signal_time_profile = signal_time_profile
        self.background_time_profile = background_time_profile
        self.source = source
        
        if infile is not None:
            indata = np.load(infile, allow_pickle = True)
            self.sob_maps = indata[0]
            self.bg_p_dec = indata[1]
        else:
            self.sob_maps = np.zeros((len(bins[0])-1, len(bins[1])-1, len(gammas)))
            self.bg_p_dec = self._create_bg_p_dec()
            for i,gamma in enumerate(tqdm(gammas)):
                self.sob_maps[:,:,i] = self._create_interpolated_ratio(gamma)
        
        if outfile is not None:
            np.save(outfile, np.array([self.sob_maps, self.bg_p_dec], dtype=object))

#-------------------------------------------------------------------------------------------------------------------------------------------------
# init helper functions
#-------------------------------------------------------------------------------------------------------------------------------------------------
    def _create_interpolated_ratio(self, gamma: np.ndarray) -> np.ndarray:
        """Generates a 2D histogram from splines of sig/bg vs dec at a range of energies.
        
        More function info...
        
        Args:
            gamma: Assumed gamma for weighting.
            
        Returns:
            Signal-over-background ratios.
        """
        # background
        bg_w = np.ones(len(self.data), dtype=float)
        bg_w /= np.sum(bg_w)
        bg_h, xedges, yedges  = np.histogram2d(np.sin(self.data['dec']),
                                               self.data['logE'],
                                               bins=self.bins,
                                               weights = bg_w)

        # signal
        sig_w = self.sim['ow'] * self.sim['trueE']**gamma
        sig_w /= np.sum(sig_w)
        sig_h, xedges, yedges = np.histogram2d(np.sin(self.sim['dec']),
                                               self.sim['logE'],
                                               bins=self.bins,
                                               weights = sig_w)

        ratio = sig_h / bg_h
        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            values = ratio[i]
            good = np.isfinite(values) & (values>0)
            x, y = self.bins[1][:-1][good], values[good]

            # Do a linear interpolation across the energy range
            spline = scipy.interpolate.UnivariateSpline(x, y, k = 1, s = 0, ext = 3)

            # And store the interpolated values
            ratio[i] = spline(self.bins[1,:-1])

        return ratio

    def _create_bg_p_dec(self, steps: int = 501) -> scipy.interpolate.UnivariateSpline:
        """Generates a spline of neutrino flux vs declination.
        
        More function info...
        
        Args:
            steps: Number of steps to bin sin(dec) into.
        
        Returns:
            A spline function representing the sin(dec) histogram.
        """
        # Our background PDF only depends on declination.
        # In order for us to capture the dec-dependent
        # behavior, we first take a look at the dec values
        # in the data. We can do this by histogramming them.
        sin_dec = np.sin(self.data['dec'])
        bins = np.linspace(-1.0, 1.0, steps)

        # Make the background histogram. Note that we do NOT
        # want to use density=True here, since that would mean
        # that our spline depends on the original bin widths!
        hist, bins = np.histogram(sin_dec,
                                  bins=bins,
                                  weights=np.ones_like(self.data['dec'])/len(self.data['dec']))

        # These values have a lot of "noise": they jump
        # up and down quite a lot. We could use fewer
        # bins, but that may hide some features that
        # we care about. We want something that captures
        # the right behavior, but is smooth and continuous.
        # The best way to do that is to use a "spline",
        # which will fit a continuous and differentiable
        # piecewise polynomial function to our data.
        # We can set a smoothing factor (s) to control
        # how smooth our spline is.
        bg_p_dec = scipy.interpolate.UnivariateSpline(bins[:-1]+np.diff(bins)/2.,
                                                      hist,
                                                      bbox=[-1.0, 1.0],
                                                      s=1.5e-5,
                                                      ext=1)
        return bg_p_dec

#-------------------------------------------------------------------------------------------------------------------------------------------------
# spatial pdfs
#-------------------------------------------------------------------------------------------------------------------------------------------------
    def _signal_pdf(self, events: np.ndarray) -> np.array:
        """Calculates the signal probability of events based on their angular distance from a source.
        
        More function info...
        
        Args:
            events: An array of events including their positional data.
            
        Returns:
            The value for the signal space pdf for the given events angular distances.
        """
        sigma = events['angErr']
        x = tools.angular_distance(events['ra'], events['dec'],
                                   self.source['ra'], self.source['dec'])
        return (1.0/(2*np.pi*sigma**2))*np.exp(-x**2/(2*sigma**2))

    def _background_pdf(self, events: np.ndarray) -> np.array:
        """Calculates the background probability of events based on their declination.
        
        More function info...
        
        Args:
            events: An array of events including their declination.
            
        Returns:
            The value for the background space pdf for the given events declinations.
        """
        background_likelihood = (1/(2*np.pi))*self.bg_p_dec(np.sin(events['dec']))
        return background_likelihood

    # signal/background
    def _evaluate_interpolated_ratios(self, events: np.ndarray) -> np.ndarray:
        """Uses calculated interpolated ratios to quickly retrieve signal/background for given events.
        
        More function info...
        
        Args:
            events: An array of events including their positional data.
        
        Returns:
            Signal-over-Background ratios at each gamma at each event location.
        """
        # Get the bin that each event belongs to
        sin_dec_idx = np.searchsorted(self.bins[0], np.sin(events['dec'])) - 1
        logE_idx = np.searchsorted(self.bins[1], events['logE'])
        
        for i in range(len(logE_idx)):
            logE_idx[i] = min(logE_idx[i], self.sob_maps.shape[1] -1)

        return self.sob_maps[sin_dec_idx, logE_idx]

    def _get_energy_splines(self, events: np.ndarray) -> np.ndarray:
        """Splines signal/background vs gamma at a set of locations using calculated interpolated ratios.
        
        More function info...
        
        Args:
            events: An array of events including their positional data.
            
        Returns:
            An array of splines of Signal-over-Background vs gamma for each event.
        """
        # Get the values for each event
        sob_ratios = self._evaluate_interpolated_ratios(events)

        # These are just values at this point. We need
        # to interpolate them in order for this to give
        # us reasonable values. Let's spline them in log-space
        sob_splines = np.zeros(len(events), dtype=object)
        for i in range(len(events)):
            spline = scipy.interpolate.UnivariateSpline(self.gammas,
                                                        np.log(sob_ratios[i]),
                                                        k = 3,
                                                        s = 0,
                                                        ext = 'raise')
            sob_splines[i] = spline
        return sob_splines

    def _get_energy_sob(self, 
                        events: np.ndarray, 
                        gamma: float, 
                        splines: scipy.interpolate.UnivariateSpline
    ) -> np.array:
        """Gets signal-over-background at given locations and gamma from calculated energy splines.
        
        More function info...
        
        Args:
            events: An array of events including their positional data.
            gamma: Some spectral index to evaluate the splines at.
            splines: Signal-over-Background splines as functions of gamma.
            
        Returns:
            An array of the spline evaluations for each event.
        """
        final_sob_ratios = np.ones_like(events, dtype=float)
        for i, spline in enumerate(splines):
            final_sob_ratios[i] = np.exp(spline(gamma))

        return final_sob_ratios

#-------------------------------------------------------------------------------------------------------------------------------------------------
# get events for trials
#-------------------------------------------------------------------------------------------------------------------------------------------------
    def _select_and_weight(self, 
                           flux_norm: float = 0, 
                           gamma: float = -2, 
                           sampling_width: float = np.radians(1)
    ) -> np.ndarray:
        """Short function info...
        
        Prunes the simulation set to only events close to a given source and calculate the
        weight for each event. Adds the weights as a new column to the simulation set.
            
        Args:
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            sampling_width: The bandwidth around the source declination to cut events.
        
        Returns:
            A reweighted simulation set around the source declination.
        """
        # Pick out only those events that are close in
        # declination. We only want to sample from those.
        sindec_dist = np.abs(self.source['dec']-self.sim['trueDec'])
        close = sindec_dist < sampling_width

        reduced_sim = rf.append_fields(self.sim[close].copy(),
                                       'weight',
                                       np.zeros(close.sum()),
                                       dtypes=np.float32)

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        reduced_sim['weight'] = reduced_sim['ow'] *\
                        flux_norm * (reduced_sim['trueE']/100.e3)**gamma

        # Apply the sampling width, which ensures that we
        # sample events from similar declinations.
        # When we do this, correct for the solid angle
        # we're including for sampling
        omega = 2*np.pi * (np.min([np.sin(self.source['dec']+sampling_width), 1]) -\
                           np.max([np.sin(self.source['dec']-sampling_width), -1]))
        reduced_sim['weight'] /= omega
        return reduced_sim

    def _inject_background_events(self) -> np.ndarray:
        """Short function info...
        
        More function info...
        
        Returns:
            An array of injected background events.
        """
        # Get the number of events we see from these runs
        n_background = self.grl['events'].sum()
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(self.data, n_background_observed).copy()

        # Assign times to our background events
        background['time'] = self.background_time_profile.random(len(background))

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2*np.pi, len(background))
        
        return background
    
    def _inject_signal_events(self, reduced_sim: np.ndarray) -> np.ndarray:
        """Short function info...
        
        More function info...
        
        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            
        Returns:
            An array of injected signal events.
        """
        # Pick the signal events
        total = reduced_sim['weight'].sum()

        n_signal_observed = scipy.stats.poisson.rvs(total)
        signal = np.random.choice(reduced_sim, n_signal_observed,
                                  p = reduced_sim['weight']/total,
                                  replace = False).copy()

        # Assign times to the signal using our time_profile class
        signal['time'] = self.signal_time_profile.random(len(signal))

        # And cut any times outside of the background range.
        bgrange = self.background_time_profile.get_range()
        contained_in_background = ((signal['time'] >= bgrange[0]) &\
                                   (signal['time'] < bgrange[1]))
        signal = signal[contained_in_background]

        # Update this number
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(signal['trueRa'], signal['trueDec'],
                                                       ones*self.source['ra'], ones*self.source['dec'],
                                                       signal['ra'], signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(signal['trueRa'], signal['trueDec'],
                                                               ones*self.source['ra'], ones*self.source['dec'],
                                                               signal['trueRa'], signal['trueDec'])
        
        return signal

#-------------------------------------------------------------------------------------------------------------------------------------------------
# calculate test statistics (user facing functions)
#-------------------------------------------------------------------------------------------------------------------------------------------------
    def produce_trial(self, 
                      reduced_sim: Optional[np.ndarray] = None, 
                      flux_norm: float = 0, 
                      gamma: float = -2, 
                      sampling_width: float = np.radians(1), 
                      random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Produces a single trial of background+signal events based on input parameters.
        
        More function info...

        Args:
            reduced_sim: Reweighted and pruned simulated events near the source declination.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A spectral index to adjust weights.
            sampling_width: The bandwidth around the source declination to cut events.
            random_seed: A seed value for the numpy RNG.
            
        Returns:
            An array of combined signal and background events.
        """
        if random_seed is not None: np.random.seed(random_seed)
        
        if reduced_sim is None: 
            reduced_sim = self._select_and_weight(flux_norm=flux_norm, 
                                                  gamma=gamma, 
                                                  sampling_width=sampling_width)

        background = self._inject_background_events()
        
        if flux_norm > 0:
            signal = self._inject_signal_events(reduced_sim)
        else:
            signal = np.empty(0, dtype=background.dtype)
        
        # Because we want to return the entire event and not just the
        # number of events, we need to do some numpy magic. Specifically,
        # we need to remove the fields in the simulated events that are
        # not present in the data events. These include the true direction,
        # energy, and 'oneweight'.
        signal = rf.drop_fields(signal, [n for n in signal.dtype.names \
                                         if not n in background.dtype.names])

        # Combine the signal background events and time-sort them.
        events = np.concatenate([background, signal])
        sorting_indices = np.argsort(events['time'])
        events = events[sorting_indices]

        # We need to check to ensure that every event is contained within
        # a good run. If the event happened when we had deadtime (when we
        # were not taking data), then we need to remove it.
        during_uptime = [np.any((self.grl['start'] <= t) & (self.grl['stop'] > t)) \
                            for t in events['time']]
        during_uptime = np.array(during_uptime, dtype=bool)
        events = events[during_uptime]

        return events

    def evaluate_ts(self,
                    events: np.ndarray,
                    ns: float = 0,
                    gamma: float = -2
    ) -> Dict:
        """Short function info...
        
        Calculates the test statistic for some collection of events at a given location
        and for some given time profiles for signal and background. Assumes gaussian
        signal profile.
            
        Args:
            events: An array of signal and background events.
            ns: A guess for the number of signal events.
            gamma: A guess for best fit spectral index of the signal.
            
        Returns:
            A dictionary of the test statistic (TS) and the best fit parameters for the TS.
        """
        # structure to store our output
        output = {
            'ts':np.nan,
            'ns':ns,
            'gamma':gamma,
            **self.signal_time_profile.default_params,
        }
        
        flux_norm = len(events)
        
        if flux_norm == 0: return output

        # Check: ns cannot be larger than flux_norm
        if ns >= flux_norm:
            ns = flux_norm - 0.00001
            
        S = self._signal_pdf(events)
        B = self._background_pdf(events)

        splines = self._get_energy_splines(events)
        t_lh_bg = self.background_time_profile.pdf(events['time'])
        
        drop = flux_norm - np.sum(S != 0)
        
        def get_ts(args):
            params = []
            ns = args[0]
            gamma = args[1]
            if len(args) > 2:
                params = args[2:]
                
            e_lh_ratio = self._get_energy_sob(events, gamma, splines)
            sig_t_pro = self.signal_time_profile.__class__(*params)
            t_lh_sig = sig_t_pro.pdf(events['time'])
            sob = S/B*e_lh_ratio * (t_lh_sig/t_lh_bg)
            ts = (ns/flux_norm*(sob - 1))+1
            return -2*(np.sum(np.log(ts)) + drop*np.log(1-ns/flux_norm))

        with np.errstate(divide='ignore', invalid='ignore'):
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters.
            x0 = [ns, gamma, *self.signal_time_profile.x0(events['time'])]
            bounds = [[0, flux_norm],
                      [-4, -1], # gamma [min, max]
                      *self.signal_time_profile.bounds(self.background_time_profile)]

            result = scipy.optimize.minimize(get_ts, x0=x0, bounds=bounds, method='L-BFGS-B')

            # Store the results in the output array
            output['ts'] = -1*result.fun
            output['ns'] = result.x[0]
            output['gamma'] = result.x[1]
            for i, key in enumerate(self.signal_time_profile.default_params):
                output[key] = result.x[2+i]

            return output

    def produce_n_trials(self, 
                         ntrials: int,
                         test_ns: float = 1,
                         test_gamma: float = -2,
                         random_seed: Optional[int] = None,
                         flux_norm: float = 0,
                         gamma: float = -2,
                         sampling_width: float = np.radians(1)
    ) -> np.ndarray:
        """Produces n trials and calculate a test statistic for each trial.
        
        More function info...
        
        Args:
            ntrials: The number of times to repeat the trial + evaluate_ts process.
            test_ns: A guess for the number of signal events.
            test_gamma: A guess for best fit spectral index of the signal.
            random_seed: A seed value for the numpy RNG.
            flux_norm: A flux normaliization to adjust weights.
            gamma: A guess for best fit spectral index of the signal.
            sampling_width: The bandwidth around the source declination to cut events.
            
        Returns:
            An array of test statistic values and their best-fit parameters for each trial.
        """
        if random_seed: np.random.seed(random_seed)

        # Cut down the sim. We're going to be using the same
        # source and weights each time, so this stops us from
        # needing to recalculate over and over again.
        reduced_sim = self._select_and_weight(flux_norm=flux_norm, gamma=gamma, sampling_width=sampling_width)

        # Build a place to store information for the trial
        dtype = np.dtype([('ts', np.float32),
                          ('ntot', np.int),
                          ('ninj', np.int),
                          ('ns', np.float32),
                          ('gamma', np.float32),
                          *self.signal_time_profile.param_dtype])
        fit_info = np.empty(ntrials, dtype=dtype)

        # We're going to cache the signal weights, which will
        # speed up our signal generation significantly.
        signal_weights = None

        for i in tqdm(range(ntrials),
                      desc=f'Running Trials (N={flux_norm:3.2e}, gamma={gamma:2.1f})',
                      unit=' trials',
                      position=0,
                      ncols = 800):

            # Produce the trial events
            trial = self.produce_trial(reduced_sim,
                                       flux_norm=flux_norm,
                                       gamma=gamma,
                                       random_seed=random_seed)

            # And get the weights
            bestfit = self.evaluate_ts(trial, ns = test_ns, gamma = test_gamma)

            fit_info['ts'][i] = bestfit['ts']
            fit_info['ntot'][i] = len(trial)
            fit_info['ninj'][i] = (trial['run'] > 200000).sum()
            fit_info['ns'][i] = bestfit['ns']
            fit_info['gamma'][i] = bestfit['gamma']
            for j, key in enumerate(self.signal_time_profile.default_params):
                fit_info[key][i] = bestfit[key]

        return fit_info

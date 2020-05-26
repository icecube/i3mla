from .. import tools
import scipy
import numpy.lib.recfunctions as rf
from ..time_profiles import gauss_profile
from tqdm.notebook import tqdm
import numpy as np

class ps_flare_llh(object):
    '''Software to perform an point-source analysis assuming some single-flaring behavior
    to the signal
    '''

    def __init__(self, data, sim, grl, gammas, bins, infile = None, outfile = None):
        '''Constructor for the class

        Arguments:
            data
            sim
            grl
            gammas
            bins
        '''
        self.data     = data
        self.sim      = sim
        self.grl      = grl
        self.bins     = bins
        self.gammas   = gammas
        
        if infile is not None:
            indata = np.load(infile, allow_pickle = True)
            self.sob_maps = indata[0]
            self.bg_p_dec = indata[1]
        else:
            self.sob_maps = np.zeros((len(bins[0])-1, len(bins[1])-1, len(gammas)))
            self.bg_p_dec = self._create_bg_p_dec()
            for i,gamma in enumerate(gammas):
                self.sob_maps[:,:,i],_ = self._create_interpolated_ratio(gamma, bins)
        
        if outfile is not None:
            np.save(outfile, np.array([self.sob_maps, self.bg_p_dec]))
        return

    # init helper functions
    def _create_interpolated_ratio(self, gamma, bins):
        '''Generate a 2D histogram from splines of signal/background vs declination at a range
            of energies.
        '''
        # background
        bg_w = np.ones(len(self.data), dtype=float)
        bg_w /= np.sum(bg_w)
        bg_h, xedges, yedges  = np.histogram2d(np.sin(self.data['dec']),
                                               self.data['logE'],
                                               bins=bins,
                                               weights = bg_w)

        # signal
        sig_w = self.sim['ow'] * self.sim['trueE']**gamma
        sig_w /= np.sum(sig_w)
        sig_h, xedges, yedges = np.histogram2d(np.sin(self.sim['dec']),
                                               self.sim['logE'],
                                               bins=bins,
                                               weights = sig_w)

        ratio = sig_h / bg_h
        for i in range(ratio.shape[0]):
            # Pick out the values we want to use.
            # We explicitly want to avoid NaNs and infinities
            values = ratio[i]
            good = np.isfinite(values) & (values>0)
            x, y = bins[1][:-1][good], values[good]

            # Do a linear interpolation across the energy range
            spline = scipy.interpolate.UnivariateSpline(x, y,
                                                        k = 1,
                                                        s = 0,
                                                        ext = 3)

            # And store the interpolated values
            ratio[i] = spline(bins[1,:-1])

        return ratio, bins

    def _create_bg_p_dec(self,):
        '''Generates a spline of neutrino flux vs declination'''
        # Our background PDF only depends on declination.
        # In order for us to capture the dec-dependent
        # behavior, we first take a look at the dec values
        # in the data. We can do this by histogramming them.
        sin_dec = np.sin(self.data['dec'])
        bins = np.linspace(-1.0, 1.0, 501)

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

    # spatial pdfs
    def _signal_pdf(self, event, source):
        '''calculate the signal probability of an event based on it's angular distance
        from a source
        '''
        sigma = event['angErr']
        x = tools.angular_distance(event['ra'], event['dec'],
                             source['ra'], source['dec'])
        return (1.0/(2*np.pi*sigma**2))*np.exp(-x**2/(2*sigma**2))

    def _background_pdf(self, event, source):
        '''calculate the background probability of an event based on it's declination
        '''
        background_likelihood = (1/(2*np.pi))*self.bg_p_dec(np.sin(event['dec']))
        return background_likelihood

    # signal/background
    def _evaluate_interpolated_ratios(self, events):
        '''Use calculated interpolated ratios to quickly retrieve signal/background for given
        events
        '''
        # Get the bin that each event belongs to
        i = np.searchsorted(self.bins[0], np.sin(events['dec'])) - 1
        j = np.searchsorted(self.bins[1], events['logE']) - 1

        return self.sob_maps[i,j]

    def _get_energy_splines(self, events):
        '''Spline signal/background vs gamma at a set of locations using calculated interpolated
        ratios
        '''
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

    def _get_energy_sob(self, events, gamma, splines):
        '''Get signal/background at given locations and gamma from calculated energy splines'''
        final_sob_ratios = np.ones_like(events, dtype=float)
        for i, spline in enumerate(splines):
            final_sob_ratios[i] = np.exp(spline(gamma))

        return final_sob_ratios

    # prune simulation around source
    def _select_and_weight(self, N=0, gamma=-2, source = {'ra':np.pi/2, 'dec':np.pi/6},
                          time_profile = None, sampling_width = np.radians(1)):
        '''Prune the simulation set to only events close to a given source and calculate the
            weight for each event. Add the weights as a new column to the simulation set
        '''
        assert('ow' in self.sim.dtype.names)
        assert(time_profile != None)

        # Pick out only those events that are close in
        # declination. We only want to sample from those.
        sindec_dist = np.abs(source['dec']-self.sim['trueDec'])
        close = sindec_dist < sampling_width

        reduced_sim = rf.append_fields(self.sim[close].copy(),
                                       'weight',
                                       np.zeros(close.sum()),
                                       dtypes=np.float32)

        # Assign the weights using the newly defined "time profile"
        # classes above. If you want to make this a more complicated
        # shape, talk to me and we can work it out.
        effective_livetime = time_profile.effective_exposure()
        reduced_sim['weight'] = reduced_sim['ow'] *\
                        N * (reduced_sim['trueE']/100.e3)**gamma *\
                        effective_livetime * 24 * 3600.

        # Apply the sampling width, which ensures that we
        # sample events from similar declinations.
        # When we do this, correct for the solid angle
        # we're including for sampling
        omega = 2*np.pi * (np.min([np.sin(source['dec']+sampling_width), 1]) -\
                           np.max([np.sin(source['dec']-sampling_width), -1]))
        reduced_sim['weight'] /= omega
        return reduced_sim

    # calculate test statistics
    def produce_trial(self, N=0, gamma=-2, source = {'ra':np.pi/2, 'dec':np.pi/6},
                      signal_time_profile = None, background_time_profile = None,
                      background_window = 14, # days
                      random_seed = None, signal_weights = None, sim = None,
                      return_signal_weights = False, sampling_width = np.radians(1)):
        '''Produces a single trial of background+signal events based on input parameters

        keyword arguments:
            N
            gamma
            source
            signal_time_profile
            background_time_profile
            sampling_width
            background_window
            random_seed
            signal_weights
            return_signal_weights
        '''
        if sim is None:
            sim = self.sim
        
        assert(background_window > 0)
        assert(signal_time_profile != None)
        assert(background_time_profile != None)

        if random_seed != None: np.random.seed(random_seed)

        # Start by calculating the background rate. For this, we'll
        # look at the number of events observed in runs just before
        # our start_time. We're picking this to exclude the start time,
        # since we don't want to include our hypothesized signal in
        # our background estimates
        start_time = background_time_profile.get_range()[0]
        fully_contained = (self.grl['start'] >= start_time-background_window) &\
                            (self.grl['stop'] < start_time)
        start_contained = (self.grl['start'] < start_time-background_window) &\
                            (self.grl['stop'] > start_time-background_window)

        background_runs = (fully_contained | start_contained)
        if not np.any(background_runs):
            print("ERROR: No runs found in GRL for calculation of "
                  "background rates!")
            raise RuntimeError
        background_grl = self.grl[background_runs]

        # Get the number of events we see from these runs and scale
        # it to the number we expect for our search livetime.
        n_background = background_grl['events'].sum()
        n_background /= background_grl['livetime'].sum()
        n_background *= background_time_profile.effective_exposure()
        n_background_observed = np.random.poisson(n_background)

        # How many events should we add in? This will now be based on the
        # total number of events actually observed during these runs
        background = np.random.choice(self.data, n_background_observed).copy()

        # Assign times to our background events
        background['time'] = background_time_profile.random(len(background))

        # Randomize the background RA
        background['ra'] = np.random.uniform(0, 2*np.pi, len(background))

        # Do we want signal events?
        reduced_sim = sim
        if N > 0:
            if "weight" not in sim.dtype.names:
                reduced_sim = self._select_and_weight(N=N,
                                                      gamma=gamma,
                                                      source = source,
                                                      time_profile = signal_time_profile,
                                                      sampling_width = sampling_width)

            # Pick the signal events
            total = reduced_sim['weight'].sum()
            n_signal_observed = scipy.stats.poisson.rvs(total)
            signal = np.random.choice(reduced_sim, n_signal_observed,
                                      p = reduced_sim['weight']/total,
                                      replace = False).copy()

            # Assign times to the signal using our time_profile class
            signal['time'] = signal_time_profile.random(len(signal))

            # And cut any times outside of the background range.
            bgrange = background_time_profile.get_range()
            contained_in_background = ((signal['time'] >= bgrange[0]) &\
                                       (signal['time'] < bgrange[1]))
            signal = signal[contained_in_background]

            # Update this number
            n_signal_observed = len(signal)

            if n_signal_observed > 0:
                ones = np.ones_like(signal['trueRa'])

                signal['ra'], signal['dec'] = tools.rotate(signal['trueRa'],
                                                           signal['trueDec'],
                                                           ones*source['ra'],
                                                           ones*source['dec'],
                                                           signal['ra'],
                                                           signal['dec'])
                signal['trueRa'], signal['trueDec'] = tools.rotate(signal['trueRa'],
                                                                   signal['trueDec'],
                                                                   ones*source['ra'],
                                                                   ones*source['dec'],
                                                                   signal['trueRa'],
                                                                   signal['trueDec'])

            # Because we want to return the entire event and not just the
            # number of events, we need to do some numpy magic. Specifically,
            # we need to remove the fields in the simulated events that are
            # not present in the data events. These include the true direction,
            # energy, and 'oneweight'.
            signal = rf.drop_fields(signal, [n for n in signal.dtype.names \
                                             if not n in background.dtype.names])
        else:
            signal = np.empty(0, dtype=background.dtype)
            signal_weights = None

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

        if return_signal_weights:
            return events, reduced_sim
        else:
            return events

    def evaluate_ts(self, events, source, bg_time_profile, sig_time_profile, ns = 0,
                    gamma = -2, minimize = True):
        '''Calculate the test statistic for some collection of events at a given location
        and for some given time profiles for signal and background
        '''
        # structure to store our output
        output = {'ts':np.nan,
                  'ns':ns,
                  'gamma':gamma,
                  't_mean':0,
                  't_sigma':0}
        N = len(events)
        if N==0:
            return output

        # Check: ns cannot be larger than N.
        if ns >= N:
            ns = N - 0.00001

        S = self._signal_pdf(events, source)
        B = self._background_pdf(events, source)

        splines = self._get_energy_splines(events)

        t_lh_bg  = bg_time_profile.pdf(events['time'])
        #t_lh_sig = sig_time_profile.pdf(events['time'])

        with np.errstate(divide='ignore', invalid='ignore'):
            # In principle, we'd want to throw everything into
            # the minimizer at once and let it work out whatever
            # correlations there are. In practice, the ns parameter
            # isn't correlated with other parameters, so it's
            # actually more efficient to fit it separately after
            # the other parameters are found. Doing it this way
            # speeds things up by a factor of ~2.5x
            def get_inner_ts(params):
                gamma, mean, sigma = params
                e_lh_ratio = self._get_energy_sob(events, gamma, splines)
                t_lh_sig = gauss_profile.gauss_profile(mean, sigma).pdf(events['time'])
                sob = S/B*e_lh_ratio * (t_lh_sig/t_lh_bg)
                ts = (1/N*(sob - 1))+1
                return -2*np.sum(np.log(ts))

            # This one will only fit ns
            def get_ts(ns, params):
                sob = params[0]
                ts = (ns/N*(sob - 1))+1
                return -2*np.sum(np.log(ts))

            if minimize:
                # Set the seed values, which tell the minimizer
                # where to start, and the bounds. First do the
                # shape parameters (just gamma, in this case).
                x0 = [gamma, np.average(events['time']), np.std(events['time'])]
                bounds = [[-4, -1], # gamma [min, max]
                          bg_time_profile.get_range(), # mean [min, max]
                          [0, bg_time_profile.effective_exposure()],
                          ]
                bf_params = scipy.optimize.minimize(get_inner_ts,
                                                    x0 = x0,
                                                    bounds = bounds,
                                                    method = 'L-BFGS-B',
                                                    )

                # and now set up the fit for ns
                x0 = [ns,]
                bounds = [[0, N],]
                e_lh_ratio = self._get_energy_sob(events, bf_params.x[0], splines)
                t_lh_sig = gauss_profile.gauss_profile(bf_params.x[1],
                                                       bf_params.x[2]).pdf(events['time'])
                sob = S/B*e_lh_ratio * (t_lh_sig/t_lh_bg)

                result = scipy.optimize.minimize(get_ts,
                                                 x0 = x0,
                                                 args = [sob,],
                                                 bounds = bounds,
                                                 method = 'L-BFGS-B',
                                                )

                # Store the results in the output array
                output['ts'] = -1*result.fun
                output['ns'] = result.x[0]
                output['gamma'] = bf_params.x[0]
                output['t_mean'] = bf_params.x[1]
                output['t_sigma'] = bf_params.x[2]

            else:
                output['ts'] = -1*get_ts(ns, [gamma,])
                output['ns'] = ns
                output['gamma'] = gamma

                # random junk
                output['t_mean'] = 0
                output['t_sigma'] = 0

            return output

    def produce_n_trials(self, ntrials,

                         # Time profiles for signal/background generation
                         background_time_profile,
                         signal_time_profile,

                         # Estimate the background rate over this many days
                         background_window = 14,

                         # Parameters to control where/when you look
                         test_ns = 1,
                         test_gamma = -2,

                         # Other
                         minimize = True,
                         random_seed = None,
                         verbose=True,
                         ncpus = 4,

                         # Signal flux parameters
                         N=0,
                         gamma=-2,
                         source = {'ra':np.pi/2, 'dec':np.pi/6},
                         sampling_width = np.radians(1)):
        '''produce n trials and calculate a test statistic for each trial'''
        if random_seed:
            np.random.seed(random_seed)

        if background_window < 1:
            print("WARN: Your window for estimating the backgroud rate is"
                  " {} and is less than 1 day. You may run into large"
                  " statistical uncertainties on the background rates which"
                  " may lead to unreliable trials. Increase your"
                  " background_window to at least a day or a week to reduce"
                  " these issues.")

        if background_time_profile.effective_exposure() > background_window:
            print("WARN: Going to estimate the background from a window"
                  " of {} days, but producing a trial of {} days. Upscaling"
                  " can be a bit dangerous, since you run the risk of missing"
                  " impacts from seasonal fluctuations. Just keep it in mind"
                  " as you run.")

        # Cut down the sim. We're going to be using the same
        # source and weights each time, so this stops us from
        # needing to recalculate over and over again.
        reduced_sim = self._select_and_weight(N = N,
                                              gamma = gamma,
                                              source = source,
                                              time_profile = signal_time_profile,
                                              sampling_width = sampling_width)

        # Build a place to store information for the trial
        dtype = np.dtype([('ts', np.float32),
                          ('ntot', np.int),
                          ('ninj', np.int),
                          ('ns', np.float32),
                          ('gamma', np.float32),
                          ('t_mean', np.float32),
                          ('t_sigma', np.float32)])
        fit_info = np.empty(ntrials, dtype=dtype)

        # We're going to cache the signal weights, which will
        # speed up our signal generation significantly.
        signal_weights = None

        for i in tqdm(range(ntrials),
                      desc='Running Trials (N={:3.2e}, gamma={:2.1f})'.format(N, gamma),
                      unit=' trials',
                      position=0,
                      ncols = 800):

            # Produce the trial events
            trial = self.produce_trial(N = N,
                                       gamma = gamma,
                                       source = source,
                                       sim = reduced_sim,
                                       background_time_profile = background_time_profile,
                                       signal_time_profile = signal_time_profile,
                                       sampling_width=sampling_width,
                                       background_window=background_window,
                                       random_seed=random_seed,
                                       signal_weights=signal_weights,
                                       return_signal_weights=False)

            # And get the weights
            bestfit = self.evaluate_ts(trial,
                                       source,
                                       bg_time_profile = background_time_profile,
                                       sig_time_profile = signal_time_profile,
                                       ns = test_ns,
                                       gamma = test_gamma,
                                       minimize = minimize)

            fit_info['ts'][i] = bestfit['ts']
            fit_info['ntot'][i] = len(trial)
            fit_info['ninj'][i] = (trial['run']>200000).sum()
            fit_info['ns'][i] = bestfit['ns']
            fit_info['gamma'][i] = bestfit['gamma']
            fit_info['t_mean'][i] = bestfit['t_mean']
            fit_info['t_sigma'][i] = bestfit['t_sigma']

        return fit_info

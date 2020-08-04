'''Spectral Modelling'''

from __future__ import print_function, division
import numpy as np
import numpy.lib.recfunctions as rf
from mla.spectral import *
from mla.timing import *
import scipy.stats
from mla import tools

class PSinjector(object):
    r'''injector of point source'''
    def __init__(self, spectrum, mc , signal_time_profile = None , background_time_profile = (0,1)):
        r'''initial the injector with a spectum and signal_time_profile. background_time_profile can be generic_profile or the time range.
        
        args:
        Spectrum: object inherited from BaseSpectrum.
        mc: Monte Carlo simulation set
        signal_time_profile(optional):Object inherited from generic_profile.Default is the same as background_time_profile.
        background_time_profile(optional):Object inherited from generic_profile.Default is a uniform_profile with time range from 0 to 1.
        '''
        self.spectrum = spectrum
        self.mc = mc
        if isinstance(background_time_profile,generic_profile):
            self.background_time_profile = background_time_profile
        else:
            self.background_time_profile = uniform_profile(background_time_profile[0],background_time_profile[1])
        if signal_time_profile == None:
            self.signal_time_profile = self.background_time_profile
        else:
            self.signal_time_profile = signal_time_profile
        return
    
    def update_spectrum(self, spectrum):
        r"""Updating the injection spectrum.
        args:
        spectrum: Object inherited from BaseSpectrum.
        """
        self.spectrum = spectrum
        return
    
    def add_background(self, background ,grl):
        r''' Add Background data into the injector such that it can also inject background 
        args:
        background: background dataset.
        grl: Good run list.
        '''
        self.background = background
        self.background_rate = len(background)/np.sum(grl['livetime'])
    
    def produce_background(self,time_window):
        r''' Samples background given the time_window'''
        n_background = self.background_rate*time_window
        n_background_observed = scipy.stats.poisson.rvs(n_background)
        background = np.random.choice(data, n_background_observed)
        background['time'] = self.background_time_profile.random(len(background))
        return background
    
    def _select_and_weight(self, ra, dec ,sampling_width = np.radians(1)):
        r'''Prune the simulation set to only events close to a given source and calculate the
            weight for each event. Add the weights as a new column to the simulation set
            args:
            ra: Right accension in radians
            dec: Declination in radians
            sampling_width(optional):The width that would be included. Default is 1 degree.
            
            return:
            The reduced Monte Carlo set with only events within the sampling width.
            
        '''
        assert('ow' in self.mc.dtype.names)

        # Pick out only those events that are close in
        # declination. We only want to sample from those.
        sindec_dist = np.abs(dec-self.mc['trueDec'])
        close = sindec_dist < sampling_width

        reduced_sim = self.mc[close].copy()

        #rescale ow
        omega = 2*np.pi * (np.min([np.sin(dec+sampling_width), 1]) -\
                           np.max([np.sin(dec-sampling_width), -1]))
        reduced_sim['ow'] /= omega
        #append weight field but only fill it with zero
        if "weight" not in reduced_sim.dtype.names:
            reduced_sim = rf.append_fields(reduced_sim.copy(),
                           'weight',
                           np.zeros(len(reduced_sim)),
                           dtypes=np.float32)
        return reduced_sim
    
    def set_source_location(self, ra, dec, sampling_width = np.radians(1)):
        r'''set the source location and select events in that dec band
            args:
            ra: Right accension in radians
            dec: Declination in radians
            sampling_width(optional):The width that would be included. Default is 1 degree.
            '''
        self.ra = ra
        self.dec = dec
        self.reduce_mc = self._select_and_weight(ra, dec, sampling_width)
    
    def sample_from_spectrum(self,seed=None,poisson=True):
        r''' Samples events from spectrum
        args:
        seed(optional): Random seed
        poisson(optional): Whether enable poisson fluctuation in number of events drawns.Default is True.
        
        returns
        Events drawn from the Monte Carlo following the spectrum.
        '''
    
        if seed != None: np.random.seed(seed)
        self.reduce_mc['weight']=self.spectrum(self.reduce_mc['trueE'])*self.reduce_mc['ow']*self.signal_time_profile.effective_exposure() * 3600 * 24

        total = self.reduce_mc['weight'].sum()
        if poisson:
            n_signal_observed = scipy.stats.poisson.rvs(total)
        else:
            n_signal_observed = int(round(total)) #round to nearest integer if no poisson fluctuation
        signal = np.random.choice(self.reduce_mc, n_signal_observed,
                                      p = self.reduce_mc['weight']/total,
                                      replace = False).copy() #Sample events
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            #Rotate events to source location
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(signal['trueRa'],
                                                       signal['trueDec'],
                                                       ones*self.ra,
                                                       ones*self.dec,
                                                       signal['ra'],
                                                       signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(signal['trueRa'],
                                                               signal['trueDec'],
                                                               ones*self.ra,
                                                               ones*self.dec,
                                                               signal['trueRa'],
                                                               signal['trueDec'])

        signal['time'] = self.signal_time_profile.random(len(signal))
        bgrange = self.background_time_profile.get_range()
        contained_in_background = ((signal['time'] >= bgrange[0]) &\
                                   (signal['time'] < bgrange[1]))
        signal = signal[contained_in_background]
        
        return signal
     
    def sample_nevents(self,n_signal_observed,seed=None):
        r''' Samples events from spectrum
        args:
        n_signal_observed: Sample size
        seed(optional): Random seed
 
        
        returns
        n_signal_observed of Events drawn from the Monte Carlo following the spectrum shape.
        '''
        if seed != None: np.random.seed(seed)
        self.reduce_mc['weight']=self.spectrum(self.reduce_mc['trueE'])*self.reduce_mc['ow']
        total = self.reduce_mc['weight'].sum()
        signal = np.random.choice(self.reduce_mc, n_signal_observed,
                                      p = self.reduce_mc['weight']/total,
                                      replace = False).copy()
        n_signal_observed = len(signal)

        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = tools.rotate(signal['trueRa'],
                                                       signal['trueDec'],
                                                       ones*self.ra,
                                                       ones*self.dec,
                                                       signal['ra'],
                                                       signal['dec'])
            signal['trueRa'], signal['trueDec'] = tools.rotate(signal['trueRa'],
                                                               signal['trueDec'],
                                                               ones*self.ra,
                                                               ones*self.dec,
                                                               signal['trueRa'],
                                                               signal['trueDec'])

        signal['time'] = self.signal_time_profile.random(len(signal))
        bgrange = self.background_time_profile.get_range()
        contained_in_background = ((signal['time'] >= bgrange[0]) &\
                                   (signal['time'] < bgrange[1]))
        signal = signal[contained_in_background]
        
        return signal
        
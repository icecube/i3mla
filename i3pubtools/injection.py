'''Spectral Modelling'''

from __future__ import print_function, division
import numpy as np
import numpy.lib.recfunctions as rf
from i3pubtools.spectral import *
from i3pubtools import tools
import scipy.stats

class PSinjector(object):
    r'''injector of point source'''
    def __init__(self, spectrum, mc):
        r'''initial the injector with a spectum'''
        self.spectrum = spectrum
        self.mc = mc
    
    def _select_and_weight(self, ra, dec ,sampling_width = np.radians(1)):
        r'''Prune the simulation set to only events close to a given source and calculate the
            weight for each event. Add the weights as a new column to the simulation set
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
        r'''set the source location and select events in that dec band'''
        self.ra = ra
        self.dec = dec
        self.reduce_mc = self._select_and_weight(ra, dec, sampling_width)
    
    def sample_from_spectrum(self,livetime,seed=None,poisson=True):
        r''' Sample events from spectrum'''
    
        if seed != None: np.random.seed(seed)
        self.reduce_mc['weight']=self.spectrum(self.reduce_mc['trueE'])*self.reduce_mc['ow']*livetime * 24 * 3600.

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

        
        return signal
     
    def sample_nevents(self,n_signal_observed,seed=None):
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

        
        return signal
        
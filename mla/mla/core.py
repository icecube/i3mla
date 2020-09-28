'''Core functionality'''

from __future__ import print_function, division
import os, sys, glob, numpy as np, matplotlib, scipy,  time
from scipy import stats, interpolate, optimize
from math import pi
import numpy.lib.recfunctions as rf
import mla
from mla.spectral import *
from mla.tools import *
from mla.timing import *
import scipy.stats
from copy import deepcopy
try:
    import cPickle as pickle
except ImportError:
    import pickle

def build_bkg_spline(data , bins=np.linspace(-1.0, 1.0, 501) , file_name = None): 
    ''' build the dec-background spline.
    args:
    bins: the sindec bins that would be used to build the histogram.
    file_name(optional): The file name of the spline saved. Default is not saving the spline.
    
    return:
    sindec-background spline.
    
    '''
    sin_dec = np.sin(data['dec'])
    
    hist, bins = np.histogram(sin_dec, 
                        bins=bins, 
                        weights=np.ones_like(data['dec'])/len(data['dec']),
                        density=True
                        )
                        
    bg_p_dec = interpolate.UnivariateSpline(bins[:-1]+np.diff(bins)/2., 
                                        hist,
                                        bbox=[-1.0, 1.0],
                                        s=1.5e-5,
                                        ext=1)
    if file_name is not None:
        with open(file_name, 'wb') as f:
            pickle.dump(bg_p_dec, f)
        
    return bg_p_dec
    
def scale_and_weight_trueDec(sim , source_dec , sampling_width = np.radians(1)):
    ''' scaling the Monte carlo using trueDec
    This is for calculating expected signal given spectrum
    args:
    sim: Monte Carlo dataset
    source_dec: Declination in radians
    sampling_width: The sampling width in rad
    
    returns:
    reduced_sim=Scaled simulation set with only events within sampling width
    '''
    sindec_dist = np.abs(source_dec-sim['trueDec'])
    
    close = sindec_dist < sampling_width
    
    reduced_sim = sim[close].copy()
    
    omega = 2*np.pi * (np.min([np.sin(source_dec+sampling_width), 1]) -\
                       np.max([np.sin(source_dec-sampling_width), -1]))             
    reduced_sim['ow'] /= omega
    
    return reduced_sim

def scale_and_weight_dec(sim , source_dec , sampling_width = np.radians(1)):
    ''' scaling the Monte carlo using dec
    This is for calculating energy S/B
    Notice that we doesn't change ow here it is unnessary
    args:
    sim: Monte Carlo dataset
    source_dec: Declination in radians
    sampling_width: The sampling width in rad
    
    returns:
    reduced_sim=Simulation set with only events within sampling width(ow unchanged)
    '''
    sindec_dist = np.abs(source_dec-sim['dec'])
    
    close = sindec_dist < sampling_width
    
    reduced_sim = sim[close].copy()
    
    return reduced_sim
    
def build_bkg_2dhistogram(data , bins=[np.linspace(-1,1,100),np.linspace(1,8,100)] , file_name = None):
    ''' build the background 2d(sindec and logE) histogram. This function a prepation for energy S/B building for custom spectrum. 
    args:
    data: Background data set
    bins: Bins defination,first one is sindec binning and the second one is logE binning.
    file_name(optional): Saving the background 2d histogram to file.Default is not saving.
    
    returns:
    bg_h,bins:The background histogram and the binning.
    '''
    bg_w=np.ones(len(data),dtype=float)
    
    bg_w/=np.sum(bg_w)
    
    bg_h,xedges,yedges=np.histogram2d(np.sin(data['dec']),data['logE'],bins=bins
                                      ,weights=bg_w)
    if file_name is not None:
        np.save(file_name,bg_h)                                
    return bg_h,bins

#The code


def create_interpolated_ratio( data, sim, gamma, bins=[np.linspace(-1,1,100),np.linspace(1,8,100)]):
    r'''create the S/B ratio 2d histogram for a given gamma.
    args:
    data: Background data
    sim: Monte Carlo Simulation dataset
    gamma: spectral index
    bins: Bins defination,first one is sindec binning and the second one is logE binning.
    
    returns:
    ratio,bins:The S/B energy histogram and the binning. 
    '''
    # background
    bins = np.array(bins)
    bg_w = np.ones(len(data), dtype=float)
    bg_w /= np.sum(bg_w)
    bg_h, xedges, yedges  = np.histogram2d(np.sin(data['dec']),
                                           data['logE'],
                                           bins=bins,
                                           weights = bg_w)
    
    # signal
    sig_w = sim['ow'] * sim['trueE']**gamma
    sig_w /= np.sum(sig_w)
    sig_h, xedges, yedges = np.histogram2d(np.sin(sim['dec']),
                                           sim['logE'],
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

    
def build_energy_2dhistogram(data, sim, bins=[np.linspace(-1,1,100),np.linspace(1,8,100)], gamma_points = np.arange(-4, -1, 0.25), file_name = None):
    ''' build the Energy SOB 2d histogram for power-law spectrum for a set of gamma.
    args:
    data: Background data
    sim: Monte Carlo Simulation dataset
    bins: Bins defination,first one is sindec binning and the second one is logE binning.
    gamma_points: array of spectral index
    
    returns:
    sob_maps,gamma_points:3d array with the first 2 axes be S/B energy histogram and the third axis be gamma_points,and the binning.        
    
    '''
    sob_maps = np.zeros((len(bins[0])-1,
                     len(bins[1])-1,
                     len(gamma_points)),
                     dtype = float)
    for i, g in enumerate(gamma_points):
        sob_maps[:,:,i], _ = create_interpolated_ratio(data, sim,g, bins )
                     
    if file_name is not None:
        np.save(file_name,sob_maps)
        np.save("gamma_point_"+file_name,gamma_points)
    return sob_maps, gamma_points


    


    
class LLH_point_source(object):
    '''The class for point source'''
    def __init__(self , ra , dec , data , sim ,  spectrum , signal_time_profile = None , background_time_profile = (0,1) , background = None, fit_position=False , bkg_bins=np.linspace(-1.0, 1.0, 501) , sampling_width = np.radians(1) , bkg_2dbins=[np.linspace(-1,1,100),np.linspace(1,8,100)] , sob_maps = None , gamma_points = np.arange(-4, -1, 0.25) ,bkg_dec_spline = None ,bkg_maps = None):
        ''' Constructor of the class
        args:
        ra: RA of the source in rad
        dec: Declination of the source in rad
        data:The data(If no background/background histogram is supplied ,it will also be used to generate background pdf)
        sim: Monte Carlo simulation
        spectrum: Spectrum , could be a BaseSpectrum object or a string name PowerLaw
        signal_time_profile: generic_profile object. This is the signal time profile.Default is the same as background_time_profile.
        background_time_profile: generic_profile object or the list of the start time and end time. This is the background time profile.Default is a (0,1) tuple which will create a uniform_profile from 0 to 1.
        background: background data that will be used to build the background dec pdf and energy S/B histogram if not supplied.Default is None(Which mean the data will be used as background.
        fit_position:Whether position is a fitting parameter. If True that it will keep all data.Default is False/
        bkg_bins: The sindec bins for background pdf(as a function of sinDec).
        sampling_width: The sampling width(in rad) for Monte Carlo simulation.Only simulation events within the sampling width will be used.Default is 1 degree.
        bkg_2dbins: The sindec and logE binning for energy S/B histogram.
        sob_maps: If the spectrum is a PowerLaw,User can supply a 3D array with sindec and logE histogram generated for different gamma.Default is None.
        gamma_points: The set of gamma for PowerLaw energy weighting.
        bkg_dec_spline: The background pdf as function of sindec if the spline already been built beforehand.
        bkg_maps: The background histogram if it is already been built(Notice it would only be needed if the spectrum is a user-defined spectrum.
        '''
        if background is None:
            self.background = data
        else:
            self.background = background
        
        try:
            self.data = rf.append_fields(data,'sindec',np.sin(data['dec']),usemask=False)#The full simulation set,this is for the overall normalization of the Energy S/B ratio
        except ValueError: #sindec already exist
            pass
                    
        self.energybins = bkg_2dbins
        self.N = len(data) #The len of the data
        self.fit_position = fit_position
        try:
            self.fullsim = rf.append_fields(sim,'sindec',np.sin(sim['dec']),usemask=False)#The full simulation set,this is for the overall normalization of the Energy S/B ratio
        except ValueError: #sindec already exist
            pass
            
        
        if isinstance(background_time_profile,generic_profile):
            self.background_time_profile = background_time_profile
        else:
            self.background_time_profile = uniform_profile(background_time_profile[0],background_time_profile[1])
        if signal_time_profile is None:
            self.signal_time_profile = deepcopy(self.background_time_profile)
        else:
            self.signal_time_profile = signal_time_profile
            
        self.sample_size = 0
        self.sampling_width = sampling_width
        
        if bkg_dec_spline is None:
            self.bkg_spline = build_bkg_spline(self.background , bins = bkg_bins)
            
        elif type(bkg_dec_spline) == str:
            with open(bkg_dec_spline, 'rb') as f:
                self.bkg_spline = pickle.load(f)
        else:
            self.bkg_spline = bkg_dec_spline
        
        if spectrum == "PowerLaw":
            self.gamma_point = gamma_points
            if sob_maps is None:
                self.ratio,self.gamma_point = build_energy_2dhistogram(self.background, sim ,bkg_2dbins ,gamma_points)
                
            elif type(sob_maps) == str:
                self.ratio = np.load(sob_maps)
                
            else:
                self.ratio = sob_maps
            self.update_position(ra,dec)
        else:
            self.spectrum = spectrum
            
            if bkg_maps is None:
                self.bg_h,self.energybins = build_bkg_2dhistogram(self.background , bins = bkg_2dbins)

            elif type(bkg_maps) == str:
                self.bg_h = np.load(bkg_maps)
                
            else:
                self.bg_h = bkg_maps
            self.update_position(ra,dec)
            self.update_energy_histogram()
            
        self.update_time_weight()
        self.update_energy_weight()
        
        return
    
    def update_position(self, ra, dec):
        r'''update the position of the point source
        args:
        ra: RA of the source in rad
        dec: Declination of the source in rad
        '''
        self.ra = ra
        self.dec = dec
        self.edge_point = (np.searchsorted(self.energybins[0],np.sin(dec-self.sampling_width))-1,np.searchsorted(self.energybins[0],np.sin(dec+self.sampling_width))-1)
        self.sim = scale_and_weight_trueDec(self.fullsim , dec , sampling_width = self.sampling_width)# Notice that this is for expected signal calculation
        self.sim_dec = scale_and_weight_dec(self.fullsim , dec , sampling_width = self.sampling_width)# This is for Energy S/B ratio calculation
        self.update_spatial()
        return
    
    

    
    def update_spatial(self):
        r'''Calculating the spatial llh and drop data with zero spatial llh'''
        signal = self.signal_pdf()
        mask = signal!=0
        if self.fit_position==False:
            self.data = self.data[mask]
            signal = signal[mask]
            self.drop = self.N - mask.sum()
        else:
            self.drop = 0
        self.spatial = signal/self.background_pdf()  
        return
        
    def update_spectrum(self,spectrum):
        r''' update the spectrum'''
        self.spectrum = spectrum
        self.update_energy_histogram()
        self.update_energy_weight()
        return
        
        
    def cut_data(self , time_range ):
        r'''Cut out data outside some time range
        args:
        time_range: array of len 2
        '''
        mask = (self.data['time']>time_range[0]) & (self.data['time']<=time_range[1])
        self.data = self.data[mask]
        self.N = len(self.data)
        self.update_spatial()
        self.update_time_weight() 
        self.update_energy_weight()
        return
        
    def update_data(self , data ):
        r'''Change the data
        args:
        data: new data
        '''
        try:
            self.data = rf.append_fields(data,'sindec',np.sin(data['dec']),usemask=False)#The full simulation set,this is for the overall normalization of the Energy S/B ratio
        except ValueError: #sindec already exist
            pass
        self.data = data
        self.N = len(data)   
        self.sample_size = 0       
        self.update_spatial()
        self.update_time_weight()
        self.update_energy_weight()
        return
        
    def signal_pdf(self):
        r'''Computer the signal spatial pdf
        return:
        Signal spatial pdf
        '''
        distance = mla.tools.angular_distance(self.data['ra'], 
                                    self.data['dec'], 
                                    self.ra, 
                                    self.dec)
        sigma = self.data['angErr']
        return (1.0)/(2*np.pi*sigma**2) * np.exp(-(distance)**2/(2*sigma**2))

    def background_pdf(self):
        r'''Computer the background spatial pdf
        return:
        background spatial pdf
        '''
        background_likelihood = (1/(2*np.pi))*self.bkg_spline(np.sin(self.data['dec']))
        return background_likelihood
    
    def update_background_time_profile(self,profile):
        r'''Update the background time profile
        args:
        profile: The background time profile(generic_profile object)
        '''
        self.background_time_profile = profile
        return
     
    def update_signal_time_profile(self,profile):
        r'''Update the signal time profile
        args:
        profile: The signal time profile(generic_profile object)
        '''
        self.signal_time_profile = profile
        return
    
    def update_time_weight(self):
        r'''Update the time weighting'''
        signal_lh_ratio = self.signal_time_profile.pdf(self.data['time'])
        background_lh_ratio = self.background_time_profile.pdf(self.data['time'])
        self.t_lh_ratio = np.nan_to_num(signal_lh_ratio/background_lh_ratio) #replace nan with zero
        return
    
    def update_energy_histogram(self):
        '''enegy weight calculation. This is slow if you choose a large sample width'''
        sig_w=self.sim_dec['ow'] * self.spectrum(self.sim_dec['trueE'])
        sig_w/=np.sum(self.fullsim['ow'] * self.spectrum(self.fullsim['trueE']))
        sig_h,xedges,yedges=np.histogram2d(self.sim_dec['sindec'],self.sim_dec['logE'],bins=self.energybins,weights=sig_w)
        with np.errstate(divide='ignore'):
            ratio=sig_h/self.bg_h
        for k in range(ratio.shape[0]):
            values=ratio[k]
            good=np.isfinite(values)&(values>0)
            x,y=self.energybins[1][:-1][good],values[good]
            if len(x) > 1:
                spline=scipy.interpolate.UnivariateSpline(x,y,k=1,s=0,ext=3)
                ratio[k]=spline(self.energybins[1][:-1])
            elif len(x)==1:
                ratio[k]=y
            else:
                ratio[k]=0
        self.ratio=ratio
        return
        
    def update_energy_weight(self, gamma = None):
        r'''
        Update the energy weight of the events.If the spectrum is user-defined one, the first part of the code will be ran and search the S/B histogram.If the spectrum is PowerLaw object, second part of the code will be ran.
        args:
        gamma: only needed if the spectrum is a PowerLaw object and you to evaluate the weight at a spectific gamma instead of optimizing the weight over gamma.
        '''
        if self.N == 0:#If no data , just do nothing
            return
            
        #First part, ran if the spectrum is user-defined
        if self.ratio.ndim == 2 :#ThreeML style 
            i = np.searchsorted(self.energybins[0],np.sin(self.data['dec']))-1
            j = np.searchsorted(self.energybins[1],self.data['logE'])-1
            i[i<self.edge_point[0]] = self.edge_point[0] #If events fall outside the sampling width, just gonna approxiamte the weight using the nearest non-zero sinDec bin.
            i[i>self.edge_point[1]] = self.edge_point[1]
            self.energy = self.ratio[i,j]
         
        #Second part, ran if the Spectrum is a PowerLaw object.
        elif self.ratio.ndim == 3: #Tradiational style with PowerLaw spectrum and spline
            sob_ratios = self.evaluate_interpolated_ratio()
            sob_spline = np.zeros(len(self.data), dtype=object)
            for i in range(len(self.data)):
                spline = scipy.interpolate.UnivariateSpline(self.gamma_point,
                                            np.log(sob_ratios[i]),
                                            k = 3,
                                            s = 0,
                                            ext = 'raise')
                sob_spline[i] = spline
            with np.errstate(divide='ignore', invalid='ignore'):
                def inner_ts(parameter):
                    gamma = parameter[0]
                    ns = parameter[1]
                    e_lh_ratio = self.get_energy_sob(gamma, sob_spline)
                    ts = ( ns/self.N * (e_lh_ratio*self.spatial*self.t_lh_ratio - 1))+1
                    return -2*(np.sum(np.log(ts))+self.drop*np.log(1-ns/self.N))
                if gamma is not None:
                    bounds= [[gamma, gamma],[0,self.N]]
                    self.gamma_best_fit = gamma
                else:
                    bounds= [[self.gamma_point[0], self.gamma_point[-1]],[0,self.N]]
                bf_params = scipy.optimize.minimize(inner_ts,
                                    x0 = [-2,1],
                                    bounds = bounds,
                                    method = 'SLSQP',
                                    )
                self.energy = self.get_energy_sob(bf_params.x[0],sob_spline)
                self.gamma_best_fit = bf_params.x[0]
                self.ns_best_fit = bf_params.x[1]
        return
    
    def get_energy_sob(self, gamma, splines):
        r'''only be used if the spectrum is PowerLaw object.
        args:
        gamma: the spectral index
        splines: the spline of S/B of each events
        
        return:
        final_sob_ratios: array of len(data) .The energy weight of each events.
        '''
        final_sob_ratios = np.ones_like(self.data, dtype=float)
        for i, spline in enumerate(splines):
            final_sob_ratios[i] = np.exp(spline(gamma))

        return final_sob_ratios
    
    def evaluate_interpolated_ratio(self):
        r'''only be used if the spectrum is PowerLaw object.Used to create the spline. Notice the self.ratio here is a 3D array with the third dimensional be gamma_points
        return:
        2D array .The energy weight of each events at each gamma point.
        '''
        i = np.searchsorted(self.energybins[0], np.sin(self.data['dec'])) - 1
        j = np.searchsorted(self.energybins[1], self.data['logE']) - 1
        return self.ratio[i,j]
    
        
    def eval_llh(self):
        r'''Calculating the llh using the spectrum'''
        if self.N == 0:
            return 0,0
        ns = (self.sim['ow'] * self.spectrum(self.sim['trueE']) * self.signal_time_profile.effective_exposure() *24*3600).sum()     
        ts =( ns/self.N * (self.energy*self.spatial*self.t_lh_ratio - 1))+1
        ts_value = 2*(np.sum(np.log(ts))+self.drop*np.log(1-ns/self.N))
        #if ts_value < 0 or np.isnan(ts_value):
        if np.isnan(ts_value) :
            ns = 0
            ts_value = 0
        return ns,ts_value
    
    def eval_llh_ns(self,ns):     
        r'''Calculating the llh with user-input ns'''
        if self.N == 0:
            return 0,0
        ts =( ns/self.N * (self.energy*self.spatial*self.t_lh_ratio - 1))+1
        ts_value = 2*(np.sum(np.log(ts))+self.drop*np.log(1-ns/self.N))
        if np.isnan(ts_value):
            ns = 0
            ts_value = 0
        return ns,ts_value
    
    def eval_llh_fit_ns(self):        
        r'''Calculating the llh with ns floating(discarded)'''
        if self.N == 0:
            return 0,0
        bounds= [[0, self.N ],]
        def get_ts(ns):   
            ts =( ns/self.N * (self.energy*self.spatial*self.t_lh_ratio - 1))+1
            return -2*(np.sum(np.log(ts))+self.drop*np.log(1-ns/self.N))
        result = scipy.optimize.minimize(get_ts,
                    x0 = [1,],
                    bounds = bounds,
                    method = 'SLSQP',
                    )
        self.fit_ns = result.x[0]
        self.fit_ts = -1*result.fun
        if np.isnan(self.fit_ts):
            self.fit_ts = 0
            self.fit_ns = 0
        return self.fit_ns,self.fit_ts
        
    def eval_llh_fit(self):        
        r'''Calculating the llh with scipy optimize result'''
        ts =( self.ns_best_fit/self.N * (self.energy*self.spatial*self.t_lh_ratio - 1))+1
        self.fit_ts = -2*(np.sum(np.log(ts))+self.drop*np.log(1-self.ns_best_fit/self.N))
        return self.ns_best_fit,self.fit_ts
    
    def get_fit_result(self):
        r'''return the fit result, Only meaningful when the spectrum is PowerLaw object.'''
        return self.gamma_best_fit,self.ns_best_fit,self.fit_ts
    
    def add_injection(self,sample):
        r'''Add injected sample
        args:
        sample: The injection sample
        '''
        self.sample_size = len(sample)+self.sample_size
        try:
            sample = rf.append_fields(sample,'sindec',np.sin(sample['dec']),usemask=False)
        except ValueError: #sindec already exist
            pass
        sample = rf.drop_fields(sample, [n for n in sample.dtype.names \
                         if not n in self.data.dtype.names])
        self.data = np.concatenate([self.data,sample])
        self.N = self.N+len(sample)
        self.update_spatial()
        self.update_time_weight()
        self.update_energy_weight()
        return
    
    def remove_injection(self,update=True):
        r'''remove injected sample
        args:
        update: Whether updating all the weighting.Default is True.
        '''
        self.data = self.data[:len(self.data)-self.sample_size]
        self.N =  self.N-self.sample_size
        self.sample_size = 0
        if update:
            self.update_spatial()
            self.update_time_weight()
            self.update_energy_weight()
        return
        
    def modify_injection(self,sample):
        r'''modify injected sample
        args:
        sample:New sample 
        '''
        self.remove_injection(update=False)
        self.add_injection(sample)
        return 
        
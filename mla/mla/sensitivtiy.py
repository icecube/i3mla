'''Core functionality'''

from __future__ import print_function, division
import os, sys, glob, numpy as np, matplotlib, scipy,  time
from scipy import stats, interpolate, optimize
from math import pi
import numpy.lib.recfunctions as rf
from mla.spectral import *
from mla.tools import *
from mla.timing import *
from mla.core import *
from mla.injection import *
import scipy.stats
from copy import deepcopy
from matplotlib import pyplot as plt, colors


class PS_sensitivity():
    def __init__(self):
        pass
        
    def background_building(self, data, sim, bkg_bins=np.linspace(-1.0, 1.0, 501), bkg_2dbins=[np.linspace(-1,1,100),np.linspace(1,8,100)],gamma_points = np.arange(-4, -1, 0.25),save_file = None):
        r''' Building the background distribution
        args:
        data:The Background
        sim: Monte Carlo simulation
        spectrum: Spectrum , could be a BaseSpectrum object or a string name PowerLaw
        bkg_2dbins: The sindec and logE binning for energy S/B histogram.
        gamma_points: The set of gamma for PowerLaw energy weighting.
        save_file: location to save the background file.Default is not saving.
        '''
        self.energybins = bkg_2dbins
        if save_file is not None:
            bkg_file = save_file + "bkg_dec.pkl"
            sob_file = save_file + "bkd_SOB.npy"
            self.bkg_spline = build_bkg_spline(data , bins = bkg_bins , file_name = bkg_file)
            self.ratio,self.gamma_points = build_energy_2dhistogram(data, sim ,bkg_2dbins ,gamma_points,file_name = sob_file)
        else:
            self.bkg_spline = build_bkg_spline(data , bins = bkg_bins )
            self.ratio,self.gamma_points = build_energy_2dhistogram(data, sim ,bkg_2dbins ,gamma_points)
        return
    
    def load_background(self, dir_name, bkg_bins=np.linspace(-1.0, 1.0, 501), bkg_2dbins=[np.linspace(-1,1,100),np.linspace(1,8,100)],gamma_points = np.arange(-4, -1, 0.25)):
        r''' Loading background distribution
        args:
        dir_name:Location to the file
        spectrum: Spectrum , could be a BaseSpectrum object or a string name PowerLaw
        bkg_2dbins: The sindec and logE binning for energy S/B histogram.
        gamma_points: The set of gamma for PowerLaw energy weighting.
        '''
        self.energybins = bkg_2dbins
        bkg_file = dir_name + "bkg_dec.pkl"
        sob_file = dir_name + "bkd_SOB.npy"
        with open(bkg_file, 'rb') as f:
                self.bkg_spline = pickle.load(f)
        self.ratio = np.load(sob_file)
        self.gamma_points = gamma_points
        return
    
    def set_point_source(self, ra , dec , data , sim ,  spectrum , signal_time_profile = None , background_time_profile = (0,1)):
        r'''Set the location of the source and load the information of the model.
        ra: RA of the source in rad
        dec: Declination of the source in rad
        data:The data
        sim: Monte Carlo simulation
        spectrum: Spectrum , could be a BaseSpectrum object or a string name PowerLaw
        signal_time_profile: generic_profile object. This is the signal time profile.Default is the same as background_time_profile.
        background_time_profile: generic_profile object or the list of the start time and end time. This is the background time profile.Default is a (0,1) tuple which will create a uniform_profile from 0 to 1.
        '''
        self.point_source=LLH_point_source(ra , dec , data , sim ,  spectrum , signal_time_profile = signal_time_profile , background_time_profile = background_time_profile,gamma_points=self.gamma_points,bkg_dec_spline=self.bkg_spline,sob_maps = self.ratio)
        self.background_time_profile = deepcopy(self.point_source.background_time_profile)
        self.signal_time_profile =  deepcopy(self.point_source.signal_time_profile)
        return
        
    def set_backround(self, background ,grl ,background_window = 14):
        r'''Setting the background information which will later be used when drawing data as background
        args:
        background:Background data
        grl:The good run list
        background_window: The time window(days) that will be used to estimated the background rate and drawn sample from.Default is 14 days
        '''
        start_time = self.background_time_profile.get_range()[0]
        fully_contained = (grl['start'] >= start_time-background_window) &\
                            (grl['stop'] < start_time)
        start_contained = (grl['start'] < start_time-background_window) &\
                            (grl['stop'] > start_time-background_window)
        background_runs = (fully_contained | start_contained)
        if not np.any(background_runs):
            print("ERROR: No runs found in GRL for calculation of "
                  "background rates!")
            raise RuntimeError
        background_grl = grl[background_runs]
            
        # Get the number of events we see from these runs and scale 
        # it to the number we expect for our search livetime.
        n_background = background_grl['events'].sum()
        n_background /= background_grl['livetime'].sum()
        n_background *= self.background_time_profile.effective_exposure()
        self.n_background = n_background
        self.background = background
        return
    
    def set_injection( self, sim , gamma = -2, signal_time_profile = None , background_time_profile = (0,1), sampling_width = np.radians(1) ,ra = None,dec = None):
        r'''Set the details of the injection.
        sim: Simulation data
        gamma: Spectral index of the injection spectrum
        signal_time_profile: generic_profile object. This is the signal time profile.Default is the same as background_time_profile.
        background_time_profile: generic_profile object or the list of the start time and end time. This is the background time profile.Default is a (0,1) tuple which will create a uniform_profile from 0 to 1.
        '''
        spectrum = PowerLaw( 100e3, 1, gamma)
        self.PS_injector = PSinjector(spectrum, sim , signal_time_profile = None , background_time_profile = background_time_profile)
        if ra is None:
            self.PS_injector.set_source_location(self.point_source.ra,self.point_source.dec,sampling_width = sampling_width)
        else:
            self.PS_injector.set_source_location(ra,dec,sampling_width = sampling_width)
        return
    
    def draw_data(self):
        r'''Draw data sample
        return:
        background: background sample
        '''
        n_background_observed = np.random.poisson(self.n_background)
        background = np.random.choice(self.background, n_background_observed).copy()
        background['time'] = self.background_time_profile.random(len(background))
        return background
    
    def draw_signal(self):
        r'''Draw signal sample
        return:
        signal: signal sample
        '''
        return self.PS_injector.sample_from_spectrum()
    
    
    
    def build_background_TS(self,n_trials = 1000):
        r'''build background TS distribution
        args:
        n_trials: Number of trials
        return:
        TS: The TS array
        '''
        TS = []
        for i in range(n_trials):
            self.point_source.update_data(self.draw_data())
            TS.append(self.point_source.eval_llh_fit_ns()[1])
        return np.array(TS)
        
    def build_signal_TS(self, signal_trials = 200 ,result = False ,result_file = None):
        r'''build signal TS distribution
        args:
        signal_trials: Number of trials
        result: Whether storing the full result in self.result.Default is False.
        result_file:Whether storing the full result in file.Default is False.
        
        return:
        TS: The TS array
        '''
        TS = []
        ts_result = []
        for i in range(signal_trials):
            data = self.draw_data()
            signal = self.draw_signal()
            signal = rf.drop_fields(signal, [n for n in signal.dtype.names \
            if not n in data.dtype.names])
            self.point_source.update_data(np.concatenate([data,signal]))
            TS.append(self.point_source.eval_llh_fit_ns()[1])
            ts_result.append(self.point_source.get_fit_result)
        if result:
            np.save(result_file, np.array(ts_result))
        return np.array(TS)
    
    def calculate_ratio_passthreshold(self,bkg_trials = 1000, signal_trials = 200 ,result = False ,result_file = None):
        r'''Calculate the ratio of signal trials passing the threshold
        args:
        bkg_trials : Number of background trials
        signal_trials: Number of signal trials
        result: Whether storing the full result in self.result.Default is False.
        result_file:Whether storing the full result in file.Default is False.
        
        return:
        result:The ratio of passing(both for three sigma and median of the background
        '''
        signal_ts = self.build_signal_TS(signal_trials ,result = result ,result_file = result_file)
        result = [(signal_ts > self.bkg_three_sigma ).sum()/float(len(signal_ts)), (signal_ts > self.bkg_median).sum()/float(len(signal_ts))]
        return result
    
    def calculate_sensitivity(self, bkg_trials = 1000, signal_trials = 200, gamma = -2, list_N = [1e-17] ,N_factor = 2 , make_plot = None ,Threshold_list=[90] , Threshold_potential = [50],result_save = False ,result_file = None):
        r'''Calculate the sensitivity plus the discovery potential
        args:
        bkg_trials : Number of background trials
        signal_trials: Number of signal trials
        gamma: Spectral index of the injection signal
        list_N:The list of flux norm to test and build the spline
        N_factor: Factor for Flux increments .If the maximum in list_N still wasn't enough to pass the threshold, the program will enter a while loop with N_factor*N tested each times until the N passed the threshold.
        make_plot: The file name of the plot saved. Default is not saving
        Threshold_list: The list of threshold of signal TS passing Median of the background TS. 
        Threshold_potential: The list of threshold of signal TS passing 3 sigma of the background TS. 
        result: Whether storing the full result in self.result.Default is False.
        result_file:Whether storing the full result in file.Default is False.

        '''
        self.Threshold_list = Threshold_list
        self.Threshold_potential = Threshold_potential
        max_threshold = np.array(Threshold_list).max()
        max_potential = np.array(Threshold_potential).max()
        list_N = np.array(deepcopy(list_N))
        result = []
        self.ts_bkg = self.build_background_TS(bkg_trials)
        self.bkg_median = np.percentile(self.ts_bkg , 50)
        self.bkg_three_sigma = np.percentile(self.ts_bkg , 99.7)
        for N in list_N:
            print("Now testing : "+ str(N))
            spectrum = PowerLaw( 100e3, N, gamma)
            self.PS_injector.update_spectrum(spectrum)
            tempresult = self.calculate_ratio_passthreshold(bkg_trials = 1000, signal_trials = 200, result = result_save ,result_file = result_file)
            print(tempresult)
            result.append(tempresult)
        if tempresult[0] < max_potential*0.01 or tempresult[1] < max_threshold*0.01:
            reach_max = False
            N = N * N_factor
            list_N = np.append(list_N,N)
        else:
            reach_max = True
        while not reach_max:
            print("Now testing : "+ str(N))
            spectrum = PowerLaw( 100e3, N, gamma)
            self.PS_injector.update_spectrum(spectrum)
            tempresult = self.calculate_ratio_passthreshold(bkg_trials = 1000, signal_trials = 200, result = result_save ,result_file = result_file)
            print(tempresult)
            result.append(tempresult)
            if tempresult[0] < max_potential*0.01 or tempresult[1] < max_threshold*0.01:
                N = N * N_factor
                list_N = np.append(list_N,N)
            else:
                reach_max = True
        result = np.array(result)
        self.result = result
        self.list_N = list_N
        self.spline_sigma = interpolate.UnivariateSpline(list_N,result[:,0] , ext = 3)
        self.spline_sen = interpolate.UnivariateSpline( list_N,result[:,1] , ext = 3)
        Threshold_result = []
        Threshold_potential_result = []
        for i in Threshold_list:
            tempspline = interpolate.UnivariateSpline(list_N,result[:,1]-i*0.01 , ext = 3)
            Threshold_result.append(tempspline.roots()[0])
            print("Threshold: " + str(i) + ", N : " + str(self.spline_sen(i*0.01)))
        for i in Threshold_potential:
            tempspline = interpolate.UnivariateSpline(list_N,result[:,0]-i*0.01 , ext = 3)
            Threshold_potential_result.append(tempspline.roots()[0])
            print("Threshold_potential: " + str(i) + ", N : " + str(self.spline_sigma(i*0.01)))    
        self.Threshold_result = Threshold_result
        self.Threshold_potential_result = Threshold_potential_result
        if make_plot != None :
           self.make_plot(make_plot)
        return
    
    def make_plot(self,file_name):
        r'''save plot to file_name
        '''
        fig, ax = plt.subplots(figsize = (12,12))
        ax.scatter(self.list_N,self.result[:,1],label = 'sensitiviy point',color='r')
        ax.scatter(self.list_N,self.result[:,0],label = 'potential point',color='b')
        ax.set_xlim(self.list_N[0],self.list_N[-1])
        ax.plot(np.linspace(self.list_N[0],self.list_N[-1],1000),self.spline_sen(np.linspace(self.list_N[0],self.list_N[-1],1000)),label = 'sensitiviy spline',color='r')
        ax.plot(np.linspace(self.list_N[0],self.list_N[-1],1000),self.spline_sigma(np.linspace(self.list_N[0],self.list_N[-1],1000)),label = 'potential spline',color='b')
        for i in range(len(self.Threshold_result)):
            ax.axvline(self.Threshold_result[i],label = 'sensitiviy '+str(self.Threshold_list[i]),color='r')
        for i in range(len(self.Threshold_potential_result)):
            ax.axvline(self.Threshold_potential_result[i],label = 'potential '+str(self.Threshold_potential[i]),color='b')
        ax.set_title("Flux norm vs passing ratio",fontsize=14)
        ax.set_xlabel(r"Flux Norm($GeV cm^{-2} s^{-1}$)",fontsize=14)
        ax.set_ylabel(r"Passing ratio",fontsize=14)
        ax.legend(fontsize=14)
        fig.savefig(file_name)
        plt.close()
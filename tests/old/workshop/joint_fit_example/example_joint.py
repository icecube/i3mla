#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import threeML
from astropy import units as u
from hawc_hal import HAL, HealpixConeROI

#new version of the plugin from Udara
from VERITASLike import VERITASLike

import os

def find_and_delete(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
             os.remove(os.path.join(root, name))

def main():
    ra, dec = 83.630, 22.020 #2HWC catalog position of Crab Nebula
    maptree = '../data/HAWC_9bin_507days_crab_data.hd5'
    response = '../data/HAWC_9bin_507days_crab_response.hd5'
    veritasdata = '../data/threemlVEGAS20hr2p45_run54809_run57993.root'
    latdirectory = '../data/lat_crab_data' # will put downloaded Fermi data there

    data_radius = 3.0
    model_radius = 8.0

    #set up HAWC dataset
    roi = HealpixConeROI(data_radius=data_radius,
                            model_radius=model_radius,
                            ra=ra,
                            dec=dec)

    hawc = HAL("HAWC", maptree, response, roi)
    hawc.set_active_measurements(1, 9) # Perform the fist only within the last nine bins
    hawc_data = {"name": "HAWC", "data":[hawc], "Emin":1e3*u.GeV, "Emax": 37e3 * u.GeV, "E0":7*u.TeV }

    #VERTIAS plugin
    with np.errstate(divide='ignore', invalid='ignore'):
        # This VERITASLike spits a lot of numpy errors. Silent them, I hope that's OK...
        # Udara told me that's normal.
        veritas = VERITASLike('veritas', veritasdata)

    veritas_data = { "name": "VERITAS", "data":[veritas], "Emin":160*u.GeV, "Emax":30e3*u.GeV, "E0":1.0*u.TeV } 

    # Fermi via Fermipy 
    tstart = '2017-01-01 00:00:00'
    tstop = '2017-03-01 00:00:00'
    evfile, scfile = threeML.download_LAT_data(ra, dec, 10.0, tstart, tstop, time_type='Gregorian', destination_directory=latdirectory)
    config = threeML.FermipyLike.get_basic_config(evfile=evfile, scfile=scfile, ra=ra, dec=dec)
    config['selection']['emax'] = 300000.0 #MeV = 300 GeV
    config['selection']['emin'] = 100.0 #MeV = 0.1 GeV 
    config['gtlike'] = {'edisp': False}
    fermi_lat = threeML.FermipyLike("LAT", config)

    lat_data = {"name":"Fermi_LAT", "data":[fermi_lat], "Emin":0.1*u.GeV, "Emax":300*u.GeV, "E0":10*u.GeV }

    # Made up "Fermi-LAT" flux points
    # Not used for now, these are just an example for how to set up XYLike data
    # XYLike points are amsumed in base units of 3ML: keV, and keV s-1 cm-2 (bug: even if you provide something else...).
    x = [ 1.38e6, 2.57e6, 4.46e6, 7.76e6, 18.19e6, 58.88e6] # keV
    y = [5.92e-14, 1.81e-14, 6.39e-15, 1.62e-15, 2.41e-16, 1.87e-17] # keV s-1 cm-2
    yerr = [1.77e-15, 5.45e-16, 8.93e-17, 4.86e-17, 5.24e-18, 7.28e-19] # keV s-1 cm-2
    # Just save a copy for later use (plot points). Will redefine similar objects with other "source_name"
    xy_test = threeML.XYLike("xy_test", x, y, yerr,  poisson_data=False, quiet=False, source_name='XY_Test')

 
    joint_data = {"name":"Fermi_VERITAS_HAWC", "data":[fermi_lat, veritas, hawc], "Emin":0.1*u.GeV, "Emax": 37e3*u.GeV, "E0":1*u.TeV}

    datasets = [veritas_data, hawc_data, lat_data, joint_data ]

    fig, ax = plt.subplots()

    #Loop through datasets and do the fit.
    for dataset in datasets:

        data = threeML.DataList(*dataset["data"])

        spectrum = threeML.Log_parabola()

        source = threeML.PointSource(dataset["name"], ra=ra, dec=dec, spectral_shape=spectrum)

        model = threeML.Model(source)
        spectrum.alpha.bounds = (-4.0, -1.0)
        spectrum.value = -2.653
        spectrum.piv.value = dataset["E0"]
        spectrum.K.value = 3.15e-22 #if not giving units, will be interpreted as (keV cm^2 s)^-1
        spectrum.K.bounds = (1e-50, 1e10)
        spectrum.beta.value = 0.15
        spectrum.beta.bounds = (-1.0, 1.0)

        model.display()

        jl = threeML.JointLikelihood(model, data, verbose=True)
        jl.set_minimizer("ROOT")
        with np.errstate(divide='ignore', invalid='ignore'):
            # This VERITASLike spits a lot of numpy errors. Silent them, I hope that's OK...
            # Udara told me that's normal.
            best_fit_parameters, likelihood_values = jl.fit()
            err = jl.get_errors()

            jl.results.write_to("likelihoodresults_{0}.fits".format(dataset["name"]), overwrite=True)
                    
        #plot spectra
        color = next(ax._get_lines.prop_cycler)['color']
        try:
            # Using a fixed version of model_plot.py
            threeML.plot_spectra(jl.results,
                ene_min=dataset["Emin"], ene_max=dataset["Emax"],
                energy_unit='GeV', flux_unit='erg/(s cm2)',
                subplot=ax,
                fit_colors=color,
                contour_colors=color,
            )
        except:
            # Using a bugged version of model_plot.py
            print('Warning: fallback without colors... Use a fixed version of model_plot.py! (3ML PR #304)')
            threeML.plot_point_source_spectra(jl.results,
                ene_min=dataset["Emin"], ene_max=dataset["Emax"],
                energy_unit='GeV', flux_unit='erg/(s cm2)',
                subplot=ax,
            )

        #workaround to get rit of gammapy temporary file
        find_and_delete("ccube.fits", "." )

    plt.xlim(5e-2, 50e3)
    plt.xlabel("Energy [GeV]" )
    plt.ylabel(r"$E^2$ dN/dE [erg/$cm^2$/s]" )
    plt.savefig("joint_spectrum.png" )


if __name__ == "__main__":

    main()

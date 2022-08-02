"""Docstring"""
# flake8: noqa
from IceCubeLike import IceCubeLike
import numpy.lib.recfunctions as rf
import numpy as np
from ... import utility_functions
from ... import time_profiles
import glob
from typing import Optional


def read(filelist: list[str]) -> np.ndarray:
    """Docstring"""
    data = []

    for f in sorted(filelist):
        x = np.load(f)
        if len(data) == 0: data = x.copy()
        else: data = np.concatenate([data, x])

    try:
        data=rf.append_fields(data, 'sindec',
                              np.sin(data['dec']),
                              usemask=False)
    except:
        pass
    return data

def create_icecubelike_from_data(
    filepath: str,
    detector_config: str,
    config: dict,
    trimsim: Optional[float],
    angerr_floor: float = np.deg2rad(0.2),
    truth: bool = False
    ) -> IceCubeLike:
    """docstring"""
    sim_files = filepath + '/' + detector_config + '*MC*npy'
    sim = np.load([i for i in glob.glob(sim_files) if "2011" not in i][0])
    if trimsim is not None:
        sim = utility_functions.trimsim(sim,trimsim)
    data_files = filepath + '/' + detector_config + '*exp*npy'
    listofdata = []
    data = read([i for i in glob.glob(data_files)])
    sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
    data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
    if not truth:
        data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
    grlfile = filepath + "/GRL/" + detector_config + "*_exp.npy"
    grl = read([i for i in glob.glob(grlfile)])
    livetime = np.sum(grl['livetime'])
    bkg_days = np.sort(grl['stop'])[-1]-np.sort(grl['start'])[0]
    background_time_profile = time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
    inject_signal_time_profile = time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
    if 'sindec' not in data.dtype.names:
    data = rf.append_fields(
        data,
        'sindec',
        np.sin(data['dec']),
        usemask=False,
    )
    if 'sindec' not in sim.dtype.names:
        sim = rf.append_fields(
            sim,
            'sindec',
            np.sin(sim['dec']),
            usemask=False,
        )
    pass
    

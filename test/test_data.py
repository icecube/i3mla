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

import numpy as np

sim_dtype = [
    ('run', '<i8'),
    ('event', '<i8'),
    ('subevent', '<i8'),
    ('ra', '<f8'),
    ('dec', '<f8'),
    ('azi', '<f8'),
    ('zen', '<f8'),
    ('time', '<f8'),
    ('trueRa', '<f8'),
    ('trueDec', '<f8'),
    ('trueE', '<f8'),
    ('logE', '<f8'),
    ('angErr', '<f8'),
    ('ow', '<f8')
]

data_dtype = [
    ('run', '<i8'),
    ('event', '<i8'),
    ('subevent', '<i8'),
    ('ra', '<f8'),
    ('dec', '<f8'),
    ('azi', '<f8'),
    ('zen', '<f8'),
    ('time', '<f8'),
    ('logE', '<f8'),
    ('angErr', '<f8')
]

grl_dtype = [
    ('run', '<i8'),
    ('start', '<f8'),
    ('stop', '<f8'),
    ('livetime', '<f8'),
    ('events', '<i8')
]

def get_random_data(length: int) -> np.ndarray:
    """generate junk data of a given length
    
    Args:
        length: length of data
        
    Returns:
        np array of type data_dtype
    """
    data = np.empty(length, dtype=data_dtype)

    data['run'] = np.random.randint(100000, high=999999, size=length)
    data['event'] = np.random.randint(1, high=9999999, size=length)
    data['subevent'] = np.random.randint(0, high=9, size=length)
    data['ra'] = np.random.random_sample(size=length) * 2 * np.pi
    data['dec'] = (np.random.random_sample(size=length) * 2 - 1) * np.pi
    data['azi'] = np.random.random_sample(size=length) * 2 * np.pi
    data['zen'] = (np.random.random_sample(size=length) * 2 - 1) * np.pi
    data['time'] = np.random.random_sample(size=length) * 10000 + 50000
    data['logE'] = np.random.random_sample(size=length) * 10
    data['angErr'] = np.random.random_sample(size=length) * .1

    return data

def get_random_sim(length: int) -> np.ndarray:
    """generate junk sim of a given length
    
    Args:
        length: length of sim
        
    Returns:
        np array of type sim_dtype
    """
    sim = np.empty(length, dtype=sim_dtype)

    sim['run'] = np.random.randint(100000, high=999999, size=length)
    sim['event'] = np.random.randint(1, high=9999999, size=length)
    sim['subevent'] = np.random.randint(0, high=9, size=length)
    sim['ra'] = np.random.random_sample(size=length) * 2 * np.pi
    sim['dec'] = (np.random.random_sample(size=length) * 2 - 1) * np.pi
    sim['azi'] = np.random.random_sample(size=length) * 2 * np.pi
    sim['zen'] = (np.random.random_sample(size=length) * 2 - 1) * np.pi
    sim['time'] = np.random.random_sample(size=length) * 10000 + 50000
    sim['trueRa'] = np.random.random_sample(size=length) * 2 * np.pi
    sim['trueDec'] = (np.random.random_sample(size=length) * 2 - 1) * np.pi
    sim['trueE'] = np.random.random_sample(size=length) * 999999899 + 100
    sim['logE'] = np.random.random_sample(size=length) * 10
    sim['angErr'] = np.random.random_sample(size=length) * .1
    sim['ow'] = np.random.random_sample(size=length)
    sim['ow'] *= np.random.random_sample(size=length) * 1e15

    return sim

def get_random_grl(data: np.ndarray) -> np.ndarray:
    """generate junk grl based on data
    
    Args:
        data: A data array to build this GRL from
        
    Returns:
        np array of type grl_dtype
    """
    runs = np.unique(data['run'])
    length = len(runs)
    grl = np.empty(length, dtype=grl_dtype)
    for i, run in enumerate(runs):
        grl[i]['run'] = run
        grl[i]['start'] = np.min(data[data['run'] == run]['time']) - .001
        grl[i]['stop'] = np.max(data[data['run'] == run]['time']) + .001
        grl[i]['livetime'] = grl[i]['stop'] - grl[i]['start']
        grl[i]['events'] = len(data[data['run'] == run])
    return grl
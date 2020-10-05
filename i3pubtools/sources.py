__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

"""A dictionary of sources and functions to convert their locations to radians"""

import numpy as np

def ra(hrs: float, mins: float, secs: float) -> float:
    """Converts right ascension to radians.
    
    More function info...
    
    Args:
        hrs: Hours.
        mins: Minutes.
        secs: Seconds.
    
    Returns:
        Radian representation of right ascension.
    """
    return (hrs*15 + mins/4 + secs/240)*np.pi/180

def dec(sign: int, deg: float, mins: float, secs: float) -> float:
    """Converts declination to radians.
    
    More function info...
    
    Args:
        sign: A positive integer for a positive sign, a negative integer for a negative sign. 
        deg: Degrees.
        mins: Minutes.
        secs: Seconds.
        
    Returns:
        Radian representation of declination.
    """
    return sign/np.abs(sign)*(deg + m/60 + s/3600)*np.pi/180

# dict of sources
sources = {'crab_nebula':{'ra':ra(5, 34, 31.94), 'dec':dec(1, 22, 0, 52.2)},
           'txs'        :{'ra':ra(5, 9, 25.9645434784), 'dec':dec(1, 5, 41, 35.333636817)},
           'm77'        :{'ra':ra(2, 42, 40.771), 'dec':dec(-1, 0, 0, 47.84)},
           'mgro1908'   :{'ra':ra(19, 7, 54), 'dec':dec(1, 6, 16, 7)},
           'sag_a_star' :{'ra':ra(17, 45, 40.0409), 'dec':dec(-1, 29, 0, 28.118)},
           'mag_l'      :{'ra':ra(5, 23, 34.5), 'dec':dec(-1, 69, 45, 22)},
           'mag_s'      :{'ra':ra(0, 52, 44.8), 'dec':dec(-1, 72, 49, 43)},
}

"""
A dictionary of sources and functions to convert their locations to radians.
"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

import numpy as np

def ra_to_rad(hrs: float, mins: float, secs: float) -> float:
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

def dec_to_rad(sign: int, deg: float, mins: float, secs: float) -> float:
    """Converts declination to radians.

    More function info...

    Args:
        sign: A positive integer for a positive sign, a negative integer for a
            negative sign.
        deg: Degrees.
        mins: Minutes.
        secs: Seconds.

    Returns:
        Radian representation of declination.
    """
    return sign/np.abs(sign)*(deg + mins/60 + secs/3600)*np.pi/180

# dict of sources
sources = {'crab_nebula':{'ra':ra_to_rad(5, 34, 31.94),
                          'dec':dec_to_rad(1, 22, 0, 52.2)},
           'txs'        :{'ra':ra_to_rad(5, 9, 25.9645434784),
                          'dec':dec_to_rad(1, 5, 41, 35.333636817)},
           'm77'        :{'ra':ra_to_rad(2, 42, 40.771),
                          'dec':dec_to_rad(-1, 0, 0, 47.84)},
           'mgro1908'   :{'ra':ra_to_rad(19, 7, 54),
                          'dec':dec_to_rad(1, 6, 16, 7)},
           'sag_a_star' :{'ra':ra_to_rad(17, 45, 40.0409),
                          'dec':dec_to_rad(-1, 29, 0, 28.118)},
           'mag_l'      :{'ra':ra_to_rad(5, 23, 34.5),
                          'dec':dec_to_rad(-1, 69, 45, 22)},
           'mag_s'      :{'ra':ra_to_rad(0, 52, 44.8),
                          'dec':dec_to_rad(-1, 72, 49, 43)},
}

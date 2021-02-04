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

from context import mla
from mla import mla

# dict of sources
sources = [
    mla.Source(
        name='crab_nebula',
        ra=mla.ra_to_rad(5, 34, 31.94),
        dec=mla.dec_to_rad(1, 22, 0, 52.2),
    ),
    mla.Source(
        name='txs',
        ra=mla.ra_to_rad(5, 9, 25.9645434784),
        dec=mla.dec_to_rad(1, 5, 41, 35.333636817),
    ),
    mla.Source(
        name='m77',
        ra=mla.ra_to_rad(2, 42, 40.771),
        dec=mla.dec_to_rad(-1, 0, 0, 47.84),
    ),
    mla.Source(
        name='mgro1908',
        ra=mla.ra_to_rad(19, 7, 54),
        dec=mla.dec_to_rad(1, 6, 16, 7),
    ),
    mla.Source(
        name='sag_a_star',
        ra=mla.ra_to_rad(17, 45, 40.0409),
        dec=mla.dec_to_rad(-1, 29, 0, 28.118),
    ),
    mla.Source(
        name='mag_l',
        ra=mla.ra_to_rad(5, 23, 34.5),
        dec=mla.dec_to_rad(-1, 69, 45, 22),
    ),
    mla.Source(
        name='mag_s',
        ra=mla.ra_to_rad(0, 52, 44.8),
        dec=mla.dec_to_rad(-1, 72, 49, 43),
    ),
]

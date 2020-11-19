__author__ = 'John Evans'
__copyright__ = ''
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

"""Functions that are generic enough to not belong in any class"""

from typing import List, Tuple

import numpy as np

def read(filelist: List[str]) -> np.ndarray:
    """Reads in and concatenate a list of numpy files.
    
    More function info...
    
    Args:
        fileList: A list of .npy file paths as strings.
        
    Returns:
        An array of data events.
    """
    data = []
    for f in sorted(filelist):
        x = np.load(f)
        if len(data) == 0: data = x.copy()
        else: data = np.concatenate([data, x])
    return data

def to_unit_vector(ra: float, dec: float) -> np.array:
    """Converts location on unit sphere to rectangular coordinates.
    
    More function info...
    
    Args:
        ra:
        dec:
    
    Returns:
        
    """
    return np.array([np.cos(ra)*np.cos(dec),
                     np.sin(ra)*np.cos(dec),
                     np.sin(dec)])

def angular_distance(ra_A: float, dec_A: float, ra_B: float, dec_B: float) -> float:
    """Calculates the angle between two points on the unit sphere.
    
    More function info...
    
    Args:
        ra_A:
        dec_A:
        ra_B:
        dec_B:
        
    Returns:
        
    """
    unit_A = to_unit_vector(ra_A, dec_A)
    unit_B = to_unit_vector(ra_B, dec_B)

    if len(unit_A.shape) != 1:
        return np.arccos(np.dot(unit_A.T, unit_B))
    else:
        return np.arccos(np.dot(unit_A, unit_B))

def cross_matrix(mat: np.ndarray) -> np.ndarray:
        """Calculate cross product matrix.
        
        A[ij] = x_i * y_j - y_i * x_j
        
        Args:
            mat: 
            
        Returns:
            
        """
        skv = np.roll(np.roll(np.diag(mat.ravel()), 1, 1), -1, 0)
        return skv - skv.T

def rotate(ra1: float, dec1: float, ra2: float, dec2: float,
           ra3: float, dec3: float) -> Tuple[float, float]:
    """Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).
    
    The rotation is performed on (ra3, dec3).
    
    Args:
        ra1:
        dec1:
        ra2:
        dec2:
        ra3:
        dec3:
    
    Returns:
        
    """
    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert(
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
    )

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) \
        + np.sin(dec1) * np.sin(dec2)

    # correct rounding errors
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1. - np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec
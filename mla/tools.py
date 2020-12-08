"""Functions that are generic enough to not belong in any class"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

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
    for filename in sorted(filelist):
        file_data = np.load(filename)
        if len(data) == 0:
            data = file_data.copy()
        else:
            data = np.concatenate([data, file_data])
    return data


def to_unit_vector(r_a: float, dec: float) -> np.array:
    """Converts location on unit sphere to rectangular coordinates.

    More function info...

    Args:
        ra:
        dec:

    Returns:

    """
    return np.array([np.cos(r_a)*np.cos(dec),
                     np.sin(r_a)*np.cos(dec),
                     np.sin(dec)])


def angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Calculates the angle between two points on the unit sphere.

    More function info...

    Args:
        ra1:
        dec1:
        ra2:
        dec2:

    Returns:

    """
    unit1 = to_unit_vector(ra1, dec1)
    unit2 = to_unit_vector(ra2, dec2)

    if len(unit1.shape) != 1:
        return np.arccos(np.dot(unit1.T, unit2))
    return np.arccos(np.dot(unit1, unit2))


def cross_matrix(mat: np.ndarray) -> np.ndarray:
    """Calculate cross product matrix.

    A[ij] = x_i * y_j - y_i * x_j

    Args:
        mat:

    Returns:

    """
    skv = np.roll(np.roll(np.diag(mat.ravel()), 1, 1), -1, 0)
    return skv - skv.T


def rotate(ra1: float, dec1: float, ra2: float, dec2: float,  # This is fine for a first release... pylint: disable=too-many-arguments, too-many-locals
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
    nTn = np.array([np.outer(nv, nv) for nv in nvec])  # This is fine for a first release... pylint: disable=invalid-name
    nx = np.array([cross_matrix(nv) for nv in nvec])  # This is fine for a first release... pylint: disable=invalid-name

    R = np.array([(1. - np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i  # This is fine for a first release... pylint: disable=invalid-name
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    r_a = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    r_a += np.where(r_a < 0., 2. * np.pi, 0.)

    return r_a, dec

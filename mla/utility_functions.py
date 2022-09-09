"""
Math functions needed for this package
"""

__author__ = "John Evans"
__copyright__ = "Copyright 2021 John Evans"
__credits__ = ["John Evans", "Jason Fan", "Michael Larson"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "John Evans"
__email__ = "john.evans@icecube.wisc.edu"
__status__ = "Development"

import numpy as np
from numba import njit


def ra_to_rad(hrs: float, mins: float, secs: float) -> float:
    """Converts right ascension to radians.

    Args:
        hrs: Hours.
        mins: Minutes.
        secs: Seconds.

    Returns:
        Radian representation of right ascension.
    """
    return (hrs * 15 + mins / 4 + secs / 240) * np.pi / 180


def dec_to_rad(sign: int, deg: float, mins: float, secs: float) -> float:
    """Converts declination to radians.

    Args:
        sign: A positive integer for a positive sign, a negative integer for a
            negative sign.
        deg: Degrees.
        mins: Minutes.
        secs: Seconds.

    Returns:
        Radian representation of declination.
    """
    return sign / np.abs(sign) * (deg + mins / 60 + secs / 3600) * np.pi / 180


def cross_matrix(mat: np.ndarray) -> np.ndarray:
    """Calculate cross product matrix.
    A[ij] = x_i * y_j - y_i * x_j
    Args:
        mat: A 2D array to take the cross product of.
    Returns:
        The cross matrix.
    """
    skv = np.roll(np.roll(np.diag(mat.ravel()), 1, 1), -1, 0)
    return skv - skv.T


def rotate(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    ra3: float,
    dec3: float,
) -> tuple:
    """Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).

    The rotation is performed on (ra3, dec3).

    Args:
        ra1: The right ascension of the point to be rotated from.
        dec1: The declination of the point to be rotated from.
        ra2: the right ascension of the point to be rotated onto.
        dec2: the declination of the point to be rotated onto.
        ra3: the right ascension of the point that will actually be rotated.
        dec3: the declination of the point that will actually be rotated.

    Returns:
        The rotated ra3 and dec3.

    Raises:
        IndexError: Arguments must all have the same dimension.
    """
    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    if not len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3):
        raise IndexError("Arguments must all have the same dimension.")

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) + np.sin(dec1) * np.sin(
        dec2
    )

    # correct rounding errors
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack(
        [np.cos(ra1) * np.cos(dec1), np.sin(ra1) * np.cos(dec1), np.sin(dec1)]
    ).T
    vec2 = np.vstack(
        [np.cos(ra2) * np.cos(dec2), np.sin(ra2) * np.cos(dec2), np.sin(dec2)]
    ).T
    vec3 = np.vstack(
        [np.cos(ra3) * np.cos(dec3), np.sin(ra3) * np.cos(dec3), np.sin(dec3)]
    ).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec ** 2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    ntn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    r = np.array(
        [
            (1.0 - np.cos(a)) * ntn_i + np.cos(a) * one + np.sin(a) * nx_i
            for a, ntn_i, nx_i in zip(alpha, ntn, nx)
        ]
    )
    vec = np.array([np.dot(r_i, vec_i.T) for r_i, vec_i in zip(r, vec3)])

    r_a = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    r_a += np.where(r_a < 0.0, 2.0 * np.pi, 0.0)

    return r_a, dec


def angular_distance(src_ra: float, src_dec: float, r_a: float, dec: float) -> float:
    """Computes angular distance between source and location.

    Args:
        src_ra: The right ascension of the first point (radians).
        src_dec: The declination of the first point (radians).
        r_a: The right ascension of the second point (radians).
        dec: The declination of the second point (radians).

    Returns:
        The distance, in radians, between the two points.
    """
    sin_dec = np.sin(dec)

    cos_dec = np.sqrt(1.0 - sin_dec ** 2)

    cos_dist = (np.cos(src_ra - r_a) * np.cos(src_dec) * cos_dec) + np.sin(
        src_dec
    ) * sin_dec
    # handle possible floating precision errors
    cos_dist = np.clip(cos_dist, -1, 1)

    return np.arccos(cos_dist)


def newton_method(sob: np.ndarray, n_drop: float) -> float:
    """Docstring

    Args:
        sob:
        n_drop:
    Returns:

    """
    newton_precision = 0
    newton_iterations = 20
    precision = newton_precision + 1
    eps = 1e-5
    k = 1 / (sob - 1)
    x = [1.0 / n_drop] * newton_iterations

    for i in range(newton_iterations - 1):
        # get next iteration and clamp
        inv_terms = x[i] + k
        inv_terms[inv_terms == 0] = eps
        terms = 1 / inv_terms
        drop_term = 1 / (x[i] - 1)
        d1 = np.sum(terms) + n_drop * drop_term
        d2 = np.sum(terms ** 2) + n_drop * drop_term ** 2
        x[i + 1] = min(1 - eps, max(0, x[i] + d1 / d2))

        if (
            x[i] == x[i + 1]
            or (x[i] < x[i + 1] and x[i + 1] <= x[i] * precision)
            or (x[i + 1] < x[i] and x[i] <= x[i + 1] * precision)
        ):
            break
    return x[i + 1]


def trimsim(sim: np.ndarray, fraction: float, scaleow: bool = True) -> np.ndarray:
    """Keep only fraction of the simulation

    Args:
        sim: simulation.
        fraction: Fraction of sim to keep(will round to int).
        scaleow: whether to scale the ow.

    Returns:
        Trimmed sim
    """
    simsize = len(sim)
    n_keep = int(fraction * simsize)
    sim = np.random.choice(sim, n_keep)
    if scaleow:
        sim["ow"] = sim["ow"] * (simsize / float(n_keep))

    return sim

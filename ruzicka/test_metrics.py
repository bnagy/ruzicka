#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

"""
Workhorse module, which contains the main distance functions
used (minmax, manhattan, and euclidean). By setting
`TARGET = gpu` below, the computation of the functions
can be accelerated on the GPU, if the `numbapro`
package is available:
    http://docs.continuum.io/numbapro/index
"""

import numba
from numpy.typing import NDArray
import numpy as np

TARGET = "cpu"


@numba.jit(nopython=True)
def minmax(x, y: NDArray[np.float64]):
    """
    Calculates the pairwise "minmax" distance between
    two vectors, but limited to the `rnd_feature_idxs`
    specified. Note that this function is symmetric,
    so that `minmax(x, y) = minmax(y, x)`.

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.

    Returns
    ----------
    float: minmax(x, y)

    References:
    ----------
    - M. Koppel and Y. Winter (2014), Determining if Two
      Documents are by the Same Author, JASIST, 65(1): 178-187.
    - Cha SH. Comprehensive Survey on Distance/Similarity Measures
      between Probability Density Functions. International Journ.
      of Math. Models and Methods in Applied Sciences. 2007;
      1(4):300–307.
    """

    # NB - vectorising this with np.minimum etc is slower.
    assert x.shape == y.shape
    mins, maxs = 0.0, 0.0
    a, b = 0.0, 0.0
    for i in range(x.shape[0]):
        a, b = x[i], y[i]

        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a

    if maxs >= 0.0:
        return 1.0 - (mins / maxs)  # avoid zero division
    return 0.0


@numba.jit(nopython=True)
def nini(x, y: NDArray[np.float64]) -> float:
    """
    Calculates the pairwise "Nini" distance between two vectors. This is defined
    as 1 - phi where phi is Pearson's Correlation applied to binary indicator
    vectors (all non-zero frequencies are converted to 1). The range of the
    distance is [0,2], and the recommended application is to large character
    n-grams (5 or more).

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.

    Returns
    ----------
    float: nini(x, y)

    References:
    ----------
    - Nini, A. (2023). A Theory of Linguistic Individuality for Authorship
      Analysis (Elements in Forensic Linguistics). Cambridge: Cambridge
      University Press. doi:10.1017/9781108974851
    """

    # The 'Nini' distance is the Pearson's-r when the vectors are converted to
    # binary indicators, i.e. any value > 0 = 1. This is the same as the cosine
    # distance for centered (binary) vectors. We do some reasonably nasty
    # stuff so we can do everything in one pass.
    xn, ny, xy, nn = 0, 0, 0, 0  # faster to keep these as ints here
    assert x.shape == y.shape
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            if y[i] > 0.0:
                xy += 1
            else:
                xn += 1
        else:  # x = 0
            if y[i] > 0.0:
                ny += 1
            else:
                nn += 1

    # means are the total bits set / vector length. Float conversion is auto in py3.
    len = xn + xy + ny + nn
    xbar = (xn + xy) / len
    ybar = (ny + xy) / len
    # Cosine is <x,y> / ||x||*||y||
    # x dot y. For every position (xn) where x is set and y isn't, those
    # positions get 1-xbar * 0-ybar, etc.
    top = (
        xn * (1.0 - xbar) * (0.0 - ybar)
        + ny * (0.0 - xbar) * (1.0 - ybar)
        + nn * (0.0 - xbar) * (0.0 - ybar)
        + xy * (1.0 - xbar) * (1.0 - ybar)
    )
    # After mean shifting, the length is x dot x, so in 1 positions, it's 1-mu
    # squared, in 0 positions it's 0-mu squared. Then sqrt each dot-product.
    bottom = math.sqrt(  # sum x_i squared
        (
            (1.0 - xbar) * (1.0 - xbar) * (xn + xy)  # 1
            + (0.0 - xbar) * (0.0 - xbar) * (ny + nn)  # 0
        )
    ) * math.sqrt(  # sum y_i squared
        ((1.0 - ybar) * (1.0 - ybar) * (ny + xy))  # 1
        + ((0.0 - ybar) * (0.0 - ybar) * (xn + nn))  # 0
    )

    if bottom==0.0:
        return 2.0
    
    return 1.0 - (top / bottom)

@numba.jit(nopython=True)
def manhattan(x, y: NDArray[np.float64]) -> float:
    """
    Calculates the conventional pairwise Manhattan city
    block distance between two vectors, but limited to
    the `rnd_feature_idxs` specified.

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.

    Returns
    ----------
    float: manhattan(x, y)

    References:
    ----------
    - Cha SH. Comprehensive Survey on Distance/Similarity Measures
      between Probability Density Functions. International Journ.
      of Math. Models and Methods in Applied Sciences. 2007;
      1(4):300–307.
    """

    diff, z = 0.0, 0.0

    assert x.shape == y.shape
    for i in range(x.shape[0]):
        z = x[i] - y[i]

        if z < 0.0:
            z = -z

        diff += z

    return diff


@numba.jit(nopython=True)
def euclidean(x, y: NDArray[np.float64]) -> float:
    """
    Calculates the conventional pairwise Euclidean
    distance between two vectors, but limited to
    the `rnd_feature_idxs` specified.

    Parameters
    ----------
    x: float array
        The first vector of the vector pair.
    y: float array
        The second vector of the vector pair.

    Returns
    ----------
    float: euclidean(x, y)

    References:
    ----------
    - Cha SH. Comprehensive Survey on Distance/Similarity Measures
      between Probability Density Functions. International Journ.
      of Math. Models and Methods in Applied Sciences. 2007;
      1(4):300–307.
    """
    diff, z = 0.0, 0.0

    assert x.shape == y.shape
    for i in range(x.shape[0]):
        z = x[i] - y[i]
        diff += z * z

    return math.sqrt(diff)


@numba.jit(nopython=True)
def common_ngrams2(x, y: NDArray[np.float64]) -> float:
    diff = 0.0

    assert x.shape == y.shape
    for i in range(x.shape[0]):
        z = 0.0

        # if x[i] > 0.0 or y[i] > 0.0: # take union ngrams (slightly better)
        # if x[i] > 0.0 or y[i] > 0.0: # take intersection ngrams
        # if x[i] > 0.0: # take intersection ngrams
        if y[i] > 0.0:  # only target text (works best):
            z = (x[i] + y[i]) / 2.0
            z = (x[i] - y[i]) / z
            diff += z * z

    return diff


@numba.jit(nopython=True)
def common_ngrams(x, y: NDArray[np.float64]) -> float:
    diff, z = 0.0, 0.0

    assert x.shape == y.shape
    for i in range(x.shape[0]):
        # if x[i] > 0.0 or y[i] > 0.0: # take union ngrams (slightly better)
        # if x[i] > 0.0 or y[i] > 0.0: # take intersection ngrams
        # if x[i] > 0.0: # take intersection ngrams
        if y[i] > 0.0:  # only target text (works best):
            z = (2.0 * (x[i] - y[i])) / (x[i] + y[i])
            diff += z * z

    return diff


@numba.jit(nopython=True)
def cosine(x, y: NDArray[np.float64]) -> float:
    numerator, denom_a, denom_b = 0.0, 0.0, 0.0

    assert x.shape == y.shape
    for i in range(x.shape[0]):
        numerator += x[i] * y[i]
        denom_a += x[i] * x[i]
        denom_b += y[i] * y[i]

    return 1.0 - (numerator / (math.sqrt(denom_a) * math.sqrt(denom_b)))

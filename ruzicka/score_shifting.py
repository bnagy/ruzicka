#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import permutations
from typing import Collection
from typing_extensions import Self
import numpy as np
import warnings

from .evaluation import c_at_1, auc
import logging

logger = logging.getLogger("ruzicka")

EPSILON = 1e-6


def rescale(value, orig_min, orig_max, new_min, new_max: float) -> float:
    """

    Rescales a `value` in the old range defined by
    `orig_min` and `orig_max`, to the new range
    `new_min` and `new_max`. Assumes that
    `orig_min` <= value <= `orig_max`.

    Parameters
    ----------
    value: float, default=None
        The value to be rescaled.
    orig_min: float, default=None
        The minimum of the original range.
    orig_max: float, default=None
        The minimum of the original range.
    new_min: float, default=None
        The minimum of the new range.
    new_max: float, default=None
        The minimum of the new range.

    Returns
    ----------
    new_value: float
        The rescaled value.

    """

    orig_span = orig_max - orig_min
    # if orig_span < 0.0 + EPSILON:
    #     raise ValueError(
    #         f"Bad span for rescale (original span {orig_min:.2f} - {orig_max:.2f})"
    #     )
    new_span = new_max - new_min
    # if new_span < 0.0 + EPSILON:
    #     raise ValueError(
    #         f"Bad span for rescale (new span {new_min:.2f} - {new_max:.2f})"
    #     )
    scaled_value = (value - orig_min) / (orig_span + EPSILON)

    return new_min + (scaled_value * new_span)


def correct_scores(
    scores: Collection[float], p1: float = 0.25, p2: float = 0.75
) -> list[float]:
    """

    Rescales a list of scores (originally between 0 and 1)
    to three new intervals:
    - a range between 0 and `p1`
    - a range between `p2`and 1
    - a range `p1` and `p2` (where every score = 0.5)

    Parameters
    ----------
    scores: array-like [nb_scores]
        The list of scores to be rescaled
    p1: float, default=.25
        The minimum of the original range.
    p2: float, default=.75
        The minimum of the original range.

    Returns
    ----------
    new_scores: floats, array-like [nb_scores]
        The rescaled scores.

    """
    if min(scores) < 0.0 - EPSILON or max(scores) > 1.0 + EPSILON:
        warnings.warn(
            "Warning: scores are expected to be in [0,1], shifting may not work properly."
        )
    new_scores = []
    for score in scores:
        if score <= p1:
            new_scores.append(rescale(score, min(scores), max(scores), 0.0, p1))
        elif score >= p2:
            new_scores.append(rescale(score, min(scores), max(scores), p2, 1.0))
        else:
            new_scores.append(0.5)
    return new_scores


class ScoreShifter:
    """

    An object to shifts the raw verification probabilities
    outputted by a system, to better account for the PAN
    metrics, which have a strict attribution threshold at 0.5.

    The shifter can be fitted on a set of train scores, by
    optimizing the threshold parameters `p1` and `p2`
    with respect to AUC x c@1, using a simple grid search.

    """

    def __init__(
        self,
        grid_size: int = 100,
        min: float = 0.2,
        max: float = 0.8,
        min_spread: float = 0.0,
    ):
        """
        Contructor.

        Parameters
        ----------
        grid_size : int = 100
            The number of points between the different values of `p1` and `p2`
            to be tested in the grid search.

        min : float = 0.2
            The minimum value allowed for p1

        max : float = 0.8
            The maximum value allowed for p2

        min_spread : float = 0.0
            The minimum distance allowed between p1 and p2. This is useful
            sometimes to stop the shifter from overfitting by optimising p1 and
            p2 in very small ranges to eliminate one or two particular false
            classifications.

        """
        self.optimal_p1: float
        self.optimal_p2: float
        self.min = min
        self.max = max
        self.min_spread = min_spread
        self.grid_size = grid_size
        self.fitted: bool = False

    def manual_fit(self, p1, p2: float) -> Self:
        """
        Manually fits the score shifter.

        Parameters
        ----------

        p1, p2: float in [0,1]
            Values between p1 and p2 will be coerced to 0.5

        Returns
        -------

        ScoreShifter

        """
        if 0 <= p1 <= p2 <= 1:
            self.optimal_p1 = p1
            self.optimal_p2 = p2
        else:
            raise ValueError("Bad values. Need 0 <= p1 <= p2 <= 1")
        self.fitted = True
        return self

    def fit(self, predicted_scores, ground_truth_scores: Collection[float]) -> Self:
        """
        Fits the score shifter on the (development) scores for
        a data set, by searching the optimal `p1` and `p2` (in terms
        of AUC x c@1) through a stepwise grid search.

        Parameters
        ----------
        prediction_scores : array [n_problems]
            The predictions outputted by a verification system.
            Assumes `0 >= prediction <=1`.

        ground_truth_scores : array [n_problems]
            The gold annotations provided for each problem.
            Will typically be `0` or `1`.

        Returns
        -------

        ScoreShifter

        """

        # define the grid to be searched:
        thresholds = np.around(
            np.linspace(self.min, self.max, num=self.grid_size, endpoint=False), 6
        )
        nb_thresholds = thresholds.shape[0]

        # intialize score containers:
        both_scores = np.zeros((nb_thresholds, nb_thresholds))
        auc_scores = both_scores.copy()
        c_at_1_scores = both_scores.copy()

        # iterate over combinations:
        gt = np.array(ground_truth_scores)
        for i, j in permutations(range(nb_thresholds), 2):
            p1, p2 = thresholds[i], thresholds[j]

            if (p1 <= p2) and (p2 - p1 >= self.min_spread):  # ensure p1 <= p2!
                corrected_scores = np.array(correct_scores(predicted_scores, p1, p2))
                auc_score = auc(corrected_scores, gt)
                c_at_1_score = c_at_1(corrected_scores, gt)
                auc_scores[i][j] = auc_score
                c_at_1_scores[i][j] = c_at_1_score
                both_scores[i][j] = auc_score * c_at_1_score

        # find 2D optimum:
        opt_p1_idx, opt_p2_idx = np.unravel_index(
            both_scores.argmax(), both_scores.shape
        )
        self.optimal_p1 = thresholds[opt_p1_idx]
        self.optimal_p2 = thresholds[opt_p2_idx]

        # print some info:
        logger.info(f"p1 for optimal combo: {self.optimal_p1:.3f}")
        logger.info(f"p2 for optimal combo: {self.optimal_p2:.3f}")
        logger.info(f"AUC for optimal combo: {auc_scores[opt_p1_idx][opt_p2_idx]:.2%}")
        logger.info(
            f"c@1 for optimal combo: {c_at_1_scores[opt_p1_idx][opt_p2_idx]:.2%}"
        )

        self.fitted = True
        return self

    def transform(self, scores: Collection[float]) -> list[float]:
        """
        Shifts the probabilities of a (new) problem series, through
        applying the score_shifter with the previously set `p1` and `p2`.

        Parameters
        ----------
        scores : array [n_problems]
            The scores to be shifted

        Returns
        ----------
        shifted_scores: floats, array-like [nb_scores]
            The shifted scores.

        """
        if not self.fitted:
            raise RuntimeError("Must fit before transforming.")

        return correct_scores(scores, p1=self.optimal_p1, p2=self.optimal_p2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Module containing the Verifier-class to perform authorship verification
in the General Imposters (GI) framework. Using sklearn-conventions,
the Verifier-object offers a generic implementation of the algorithm
described in e.g.:
  - M. Koppel and Y. Winter (2014), Determining if Two Documents are
    by the Same Author, JASIST, 65(1): 178-187
  - Stover, J. A. , Y. Winter, M. Koppel, M. Kestemont (2015).
    Computational Authorship Verification Method Attributes New Work
    to Major 2nd Century African Author, JASIST,  doi: 10.1002/asi.23460.
  - ...

"""

import logging
import heapq

import numpy as np
import scipy as sp
import pandas as pd
import numpy.typing as npt
from typing import Collection, Callable

# import the pairwise distance functions:
from .test_metrics import minmax, manhattan, euclidean, common_ngrams, cosine, nini

CPU_METRICS: dict[str, Callable] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "minmax": minmax,
    "cng": common_ngrams,
    "cosine": cosine,
    "nini": nini,
}

logger = logging.getLogger("ruzicka")


class BDIVerifier:
    """

    Offers a generic implementation a generic implementation
    of the authorship verification algorithm described in e.g.:
      - M. Koppel and Y. Winter (2014), Determining if Two Documents are
        by the Same Author, JASIST, 65(1): 178-187
      - Stover, J. A. , Y. Winter, M. Koppel, M. Kestemont (2015).
        Computational Authorship Verification Method Attributes New Work
        to Major 2nd Century African Author, JASIST,  doi: 10.1002/asi.23460.
      - ...

    The object follow sklearn-like conventions, offering `fit()` and
    `predict_proba()`.

    """

    def __init__(
        self,
        metric: str = "manhattan",
        method: str = "ranked",
        nb_bootstrap_iter: int = 100,
        random_state: int = 1066,
        rnd_prop: float = 0.35,
        balance: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        metric: str
            The distance metric used; should be one of:
                + minmax
                + manhattan
                + euclidean
                + cosine
                + cng
                + nini

        nb_bootstrap_iter: int, default = 100
            Indicates the number of bootstrap iterations to be used (e.g. 100).
            If this evaluates to False, we run a naive version of the imposter
            algorithm without bootstrapping; i.e. we simply check once whether
            the target author appears to be a test document's nearest neighbour
            among the imposters).

        random_seed: int, default = 1066
            Integer used for seeding the random streams.

        rnd_prop: scalar, default = 0.35
            Float specifying the number of features to be randomly sampled in
            each iteration.
        """

        # some sanity checks:
        assert (rnd_prop > 0.0) and (rnd_prop < 1.0)
        if not method in ["ranked", "random", "closest"]:
            raise ValueError(
                f"Unsupported method {self.method}, valid are: random, closest, ranked"
            )
        # set rnd seeds:
        self.rnd = np.random.default_rng(seed=random_state)

        self.method = method
        self.nb_bootstrap_iter = nb_bootstrap_iter
        self.rnd_prop = rnd_prop
        self.train_X: npt.NDArray
        self.train_y: npt.NDArray
        self.balance = balance
        self.fitted: bool = False

        try:
            self.metric_fn = CPU_METRICS[metric]
        except KeyError:
            raise ValueError(
                f"Unknown metric {metric}. Valid are: {list(CPU_METRICS.keys())}"
            )

    def fit(self, X: Collection[Collection[float]], y: Collection[int]):
        """
        Runs very light, memory-based like fitting Method
        which primarily stores `X` and `y` in memory.

        Parameters
        ----------
        X: floats, array-like [nb_documents, nb_features]
            The 2D matrix representing the training instance-based
            to be memorized.

        y, array of ints [nb_documents]
            An int-encoded representation of the correct authorship
            for each training documents.

        """

        logger.info(f"Fitting on {len(y)} documents...")
        self.train_X = np.array(X, dtype="float")
        self.train_y = np.array(y, dtype="int")

        self.fitted = True

    def _balanced_subsample(
        self, y: pd.Series, size: int = 0, rng: np.random.Generator = None
    ) -> list[int]:
        """
        From a list of classes (as ints), return a random subsample that is
        balanced down to the number of the smallest class (if the smallest class
        has 10 members, it will sample 10 entries from every class) by default,
        or to `size` samples if given.

        The pattern is to sample the indices from your data and then use the
        sample indices here to subset the rows (elsewhere)

        Parameters
        ----------

        y: pd.Series[int]
            indicies to sample from

        size: int, default=0
            number of samples to take from each class. If size is 0 then the
            size of the smallest class is used.

        rng: np.random.Generator default=self.rnd
            rng to use for sample

        Returns
        -------

        list[int] (size * classes, )
            The sample indices
        """
        subsample = []

        if rng is None:
            rng = self.rnd
        if size == 0:
            n_smp = y.value_counts().min()
        else:
            n_smp = int(size / len(y.value_counts().index))

        for label in y.value_counts().index:
            samples = y[y == label].index.values
            index_range = range(samples.shape[0])
            indexes = rng.choice(index_range, size=n_smp, replace=False)
            subsample += samples[indexes].tolist()

        return subsample

    def _bootstrap_imposters(
        self, test_vec: npt.NDArray[np.float64], target_int: int, nb_imposters: int
    ) -> list[float]:
        """
        Run the bootstrap distance imposters algorithm. Many parameters that
        affect the operation here are set in the constructor. This routine works
        on one single sample, so it's called in a loop in `predict_proba`

        Parameters
        ----------

        test_vec: NDArray[float64]
            The vector to examine

        target_int: int
            The candidate to consider from the corpus array

        nb_imposters: int
            The number of imposters from the corpus array to examine

        Returns
        -------

        list[float] (self.nb_bootstrap_iter,)
            The distance recorded for each bootstrap iteration
        """
        # X at the row indices where y matches the condition
        candidates = self.train_X[(self.train_y == target_int).nonzero()]
        others = self.train_X[(self.train_y != target_int).nonzero()]
        differences: list[float] = []
        cand_samps: npt.NDArray[np.float64] = []
        other_samps: npt.NDArray[np.float64] = []
        pdy = pd.Series(self.train_y)
        if self.method == "random":
            # choose n random row indices with replacement, all columns. This will
            # still work if n > num_candidates because it will oversample.
            cand_samps = candidates[
                self.rnd.choice(
                    candidates.shape[0], self.nb_bootstrap_iter, replace=True
                ),
                :,
            ]
            other_samps = others[
                self.rnd.choice(others.shape[0], self.nb_bootstrap_iter, replace=True),
                :,
            ]

        # At each bootstrap iteration we choose a different feature subset
        for i in range(self.nb_bootstrap_iter * 2):
            if self.balance:
                # Balances the sample down to the least abundant class
                ss = self._balanced_subsample(pdy, rng=self.rnd)  # indices into y
                candidates = self.train_X[np.where(self.train_y[ss] == target_int)]
                others = self.train_X[np.where(self.train_y[ss] != target_int)]
            try:
                # This try is to catch subsamples from sparse data where some column
                # subsets are just empty and so the distance metrics yield divzero.
                # We try for n*2 iterations, returning early when we have n and
                # raising if we don't get there.

                # from 1d vectors, choose (self.rnd_prop * width_of_X) random column indices (no
                # replacement)
                ridx = self.rnd.choice(
                    self.train_X.shape[1],
                    int(self.train_X.shape[1] * self.rnd_prop),
                    replace=False,
                )

                # compare the test vector to one in-sample and one outsample (with
                # bootstrap columns), then record the difference of distances
                if self.method == "random":
                    in_dist = self.metric_fn(test_vec[ridx], cand_samps[i][ridx])
                    out_dist = self.metric_fn(test_vec[ridx], other_samps[i][ridx])
                    differences.append(out_dist - in_dist)

                # compare the test vector to the closest in-sample and out-sample, then
                # record the difference of distances (like vanilla Kestemont GI)
                elif self.method == "closest":
                    in_dists = [
                        self.metric_fn(test_vec[ridx], cand_samp[ridx])
                        for cand_samp in candidates
                    ]
                    out_dists = [
                        self.metric_fn(test_vec[ridx], other_samp[ridx])
                        for other_samp in others
                    ]
                    differences.append(min(out_dists) - min(in_dists))

                # compare the test vector to the closest in-sample and out-sample, then
                # record the scaled difference of distances for the smallest 3 (like
                # Kestemont GI with Eder Boostrap Consensus Tree stye ranking)
                elif self.method == "ranked":
                    in_dists = [
                        self.metric_fn(test_vec[ridx], cand_samp[ridx])
                        for cand_samp in candidates
                    ]
                    if nb_imposters > 0:
                        # if they've given us too many imposters this will
                        # raise, which is as informative as any check I could do
                        # here
                        this_others = others[
                            self.rnd.choice(
                                others.shape[0], nb_imposters, replace=False
                            ),
                            :,
                        ]
                    else:
                        this_others = others

                    out_dists = [
                        self.metric_fn(test_vec[ridx], other_samp[ridx])
                        for other_samp in this_others
                    ]

                    # faster than sorting and slicing
                    top_in = heapq.nsmallest(3, in_dists)
                    top_out = heapq.nsmallest(3, out_dists)
                    d = 0
                    for nn in range(3):
                        # smallest distances are unscaled, seccond is halved, etc
                        try:
                            d += (top_out[nn] - top_in[nn]) / (nn + 1)
                        except IndexError:
                            # if there are fewer than three candidates re-use
                            # the last (sorted) distance as a dummy value so the
                            # final distances are as comparable as we can make
                            # them. If we abort early then those entries would
                            # have weirdly low distances.
                            d += (top_out[-1] - top_in[-1]) / (nn + 1)

                    differences.append(d)
                else:
                    # should be impossible since we check in the constructor
                    raise ValueError(
                        f"Unsupported method {self.method}, valid are: random, closest, ranked"
                    )
            except ZeroDivisionError:
                continue
            if len(differences) >= self.nb_bootstrap_iter:
                return differences[: self.nb_bootstrap_iter]
        raise ValueError("Too many ZeroDivisionErrors. Data too sparse?")

    def predict_proba(
        self,
        test_X: Collection[Collection[float]],
        test_y: Collection[int],
        nb_imposters: int = 30,
    ) -> npt.NDArray[np.float64]:
        """

        Given a `test_vector` and an integer representing a target authors
        (`target_int`), we retrieve the distance to the nearest document in the
        training data, which is NOT authored by the target author. In the
        distance calculation, we only take into account the feature values
        specified in `rnd_feature_idxs` (if the latter parameter is specified);
        else, we use the entire feature space. Note that we each time sample a
        random number of imposters from the available training documents, the
        number of which is specified by `nb_imposters`.

        We apply the normal verification method, using self.nb_bootstrap_iter
        iterations. In this case, the returned probabilities represent the
        proportions of bootstraps in which the target_author yieled a closer
        neighbour, than any other of the randomly sampled imposter authors.

        Parameters
        ----------
        test_X : floats, array-like [nb_test_problems, nb_features]
            A 2D matrix representing the test documents in vectorized format.
            Should not contain documents that are also present in the training
            data.

        test_y : list of ints [nb_test_problems]
            An int encoding the target_authors for each test problem. These are
            not necessarily the correct author for the test documents; only the
            authors against which the authorship of the individual
            test_documents has to be verified. All authors in test_y *must* be
            present in the training data.

        nb_imposters : int, default=30
            Specifies the number of imposter or distractor documents which are
            randomly sampled from the training documents which were not written
            by the target author. Use -1 to consider all imposters at each step.

        Returns
        ----------
        probas : NDArray[float64] (num_problems,)
            A score assigned to each individual verification problem, indicating
            the likelihood with which the verifier would attrribute `test_X[i]`
            to candidate author `test[i]`.

        Attributes
        ----------

        After the run, the full set of distance arrays for each sample are
        available in `_dist_arrays`

        """
        if not self.fitted:
            raise RuntimeError("Cannot predict without training.")

        dist_arrays = []
        logger.info(f"Predicting on {len(test_y)} documents")
        # we accept collections but coerce to NDArray internally
        for vec, candidate_int in zip(np.array(test_X), np.array(test_y)):
            dist_arrays.append(
                self._bootstrap_imposters(vec, candidate_int, nb_imposters)
            )
        self._dist_arrays = dist_arrays
        probas = [(100 - sp.stats.percentileofscore(x, 0)) / 100.0 for x in dist_arrays]
        return np.array(probas, dtype="float64")

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
import numba

import numpy as np
import numpy.typing as npt
from typing import Collection, Callable

from sklearn.neighbors import NearestCentroid

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


class Order2Verifier:
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
        base: str = "profile",
        rank: bool = True,
        nb_bootstrap_iter: int = 0,
        random_state: int = 1066,
        rnd_prop: float = 0.5,
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
                + nini

        base, str
            Indicates whether to use an instance-based or profile-based approach
            for each author; should be 'profile' or 'instance'.

        rank, str, default=True
            If True, use Ranking-based Imposters (RBI) from Potha, N.,
            Stamatatos, E. Improved algorithms for extrinsic author
            verification. Knowl Inf Syst 62, 1903–1921 (2020). Does NOT
            implement the 'most relevant imposters' approach (just bootstrap
            harder!).

        nb_bootstrap_iter: int, default=100
            Indicates the number of bootstrap iterations to be used (e.g. 100).
            If this evaluates to False, we run a naive version of the imposter
            algorithm without bootstrapping; i.e. we simply check once whether
            the target author appears to be a test document's nearest neighbour
            among the imposters).

        random_state: int, default=1066
            Integer used for seeding the random streams.

        rnd_prop: scalar, default=.5
            Float specifying the number of features to be randomly sampled in
            each iteration.
        """

        # some sanity checks:
        assert base in ("profile", "instance")
        assert (rnd_prop > 0.0) and (rnd_prop < 1.0)

        # set rnd seeds:
        self.rnd = np.random.RandomState(seed=random_state)

        self.base = base
        self.rank = rank
        self.nb_bootstrap_iter = nb_bootstrap_iter
        self.rnd_prop = rnd_prop
        self.train_X: npt.NDArray
        self.train_y: npt.NDArray
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
        which primarily stores `X` and `y` in memory. In the
        case of profile-based verifier, we store a single,
        mean centroid per author in memory.

        Parameters
        ----------
        X: floats, array-like [nb_documents, nb_features]
            The 2D matrix representing the training instance-based
            to be memorized.

        y, array of ints [nb_documents]
            An int-encoded representation of the correct authorship
            for each training documents.

        References
        ----------
        - Daelemans, W. & van den Bosch, A. (2005). Memory-Based
          Language Processing. Cambridge University Press.
        - M. Koppel and S. Seidman (2013), Automatically
          Identifying Pseudepigraphic Texts, EMNLP-13: 1449-1454.

        """

        logger.info(f"Fitting on {len(y)} documents in {self.base} mode...")
        if self.base == "instance":
            self.train_X = np.array(X, dtype="float")
            self.train_y = np.array(y, dtype="int")

        elif self.base == "profile":
            self.train_X = np.array(
                NearestCentroid().fit(np.array(X), np.array(y)).centroids_,
                dtype="float",
            )  # mean centroids
            self.train_y = np.array(range(self.train_X.shape[0]), dtype="int")

        self.fitted = True

    def _dist_closest_target(
        self,
        test_vector: npt.NDArray[np.float64],
        target_int: int,
        rnd_feature_idxs: npt.NDArray[np.int32] = np.array([], dtype="int32"),
    ) -> float:
        """

        Given a `test_vector` and an integer representing a target
        authors (`target_int`), we retrieve the distance to the
        nearest document by the target author in the training data.
        In the distance calculation, we only take into account
        the feature values specified in `rnd_feature_idxs` (if
        the latter parameter is specified); else, we use the
        entire feature space.

        Parameters
        ----------
        test_vector : array of floats [nb_features]
            The 1D vector representing the vectorized test document
            for a particular problems.

        target_int : int
            An int encoding the target_author against which the
            authorship of `test_vector` has to be verified. This
            author is assumed to have at least one document in the
            training data.

        rnd_feature_idxs : list of ints, default='all'
            Integer list, specifying the indices of the feature
            values which are should in the distance calculation.
            If unspecified, we use the entire feature space.

        Returns
        ----------
        dist : float
            The actual distance to the nearest document vector
            in memory by the target author.

        """

        # use entire feature space if necessary:
        if rnd_feature_idxs.size == 0:  # use entire feature space
            rnd_feature_idxs = np.array(range(test_vector.size), dtype="int")

        # calculate distance to nearest neighbour for the
        # target author (which potentially has only 1 item):
        min_dist = float("inf")
        tv = test_vector[rnd_feature_idxs]
        for idx in range(self.train_y.size):
            if self.train_y[idx] == target_int:
                d = self.metric_fn(self.train_X[idx][rnd_feature_idxs], tv)
                if d < min_dist:
                    min_dist = d

        return min_dist

    def _dist_non_targets(
        self,
        test_vector: npt.NDArray[np.float64],
        target_int: int,
        nb_imposters: int = 0,
        rnd_feature_idxs: npt.NDArray[np.int32] = np.array([], dtype="int32"),
    ) -> npt.NDArray[np.float64]:
        """

        Given a `test_vector` and an integer representing a target
        authors (`target_int`), we retrieve the distances to the
        documents in the training data, which are NOT authored
        by the target author. In the distance calculation, we only
        take into account the feature values specified in
        `rnd_feature_idxs` (if the latter parameter is specified);
        else, we use the entire feature space. Note that we each time
        sample a random number of imposters from the available training
        documents, the number of which is specified by `nb_imposters`.

        Parameters
        ----------
        test_vector : array of floats [nb_features]
            The 1D vector representing the vectorized test document
            for a particular problems.

        target_int : int
            An int encoding the target_author against which the
            authorship of `test_vector` has to be verified. This
            author is assumed to have at least one document in the
            training data.

        nb_imposters : int, default=0
            Specifies the number of imposter or distractor documents
            which are randomly sampled from the training documents
            which were not written by the target author.

        rnd_feature_idxs : list of ints, default=[]
            Integer list, specifying the indices of the feature
            values which are should in the distance calculation.
            If empty, we use the entire feature space.

        Returns
        ----------
        dists : np.NDArray[float64]
            The actual distances to the document vectors
            in memory, which were not written by the target author,
            among a number of randomly sampled imposter documents.

        """

        # use entire feature space if necessary:
        if rnd_feature_idxs.size == 0:
            rnd_feature_idxs = np.array(range(test_vector.size), dtype="int")

        # calculate distance to nearest neighbour for any
        # author whom is NOT the target author
        non_target_idxs = [
            i for i in range(self.train_y.size) if self.train_y[i] != target_int
        ]

        # randomly pick a subset of imposters:
        self.rnd.shuffle(non_target_idxs)
        dists = np.zeros(len(non_target_idxs[:nb_imposters]), dtype=np.float64)
        tv = test_vector[rnd_feature_idxs]
        for i, idx in enumerate(non_target_idxs[:nb_imposters]):
            dists[i] = self.metric_fn(self.train_X[idx][rnd_feature_idxs], tv)
        return np.sort(dists)

    def predict_proba(
        self,
        test_X: Collection[Collection[float]],
        test_y: Collection[int],
        nb_imposters: int = 0,
    ) -> npt.NDArray[np.float64]:
        """

        Given a `test_vector` and an integer representing a target
        authors (`target_int`), we retrieve the distance to the
        nearest document in the training data, which is NOT authored
        by the target author. In the distance calculation, we only
        take into account the feature values specified in
        `rnd_feature_idxs` (if the latter parameter is specified);
        else, we use the entire feature space. Note that we each time
        sample a random number of imposters from the available training
        documents, the number of which is specified by `nb_imposters`.

        Two routines are distinguished:
        - If self.nb_bootstrap_iter evaluates to `False`, we run a
          naive version of the imposter algorithm without boot-
          strapping; i.e. we simply check once whether the target
          author appears to be a test document's nearest neighbour
          among all imposters, using all features available. In this
          case, all probabilities returned will be `0` or `1`.
        - Else, we apply the normal verification method, using
          self.nb_bootstrap_iter iterations. In this case, the re-
          turned probabilities represent the proportions of bootstraps
          in which the target_author yieled a closer neighbour, than
          any other of the randomly sampled imposter authors.

        Parameters
        ----------
        test_X : floats, array-like [nb_test_problems, nb_features]
            A 2D matrix representing the test documents in vectorized
            format. Should not contain documents that are also present
            in the training data.

        test_y : list of ints [nb_test_problems]
            An int encoding the target_authors for each test problem.
            These are not necessarily the correct author for the test
            documents; only the authors against which the authorship
            of the individual test_documents has to be verified. All
            authors in test_y *must* be present in the training data.

        nb_imposters : int, default=0
            Specifies the number of imposter or distractor documents
            which are randomly sampled from the training documents
            which were not written by the target author.

        Returns
        ----------
        probas : list of floats, array-like [nb_problems]
            A score assigned to each individual verificatio
            problem, indicating the likelihood with which
            the verifier would attrribute `test_X[i]` to
            candidate author `test[i]`.

        Note
        ----------
        It is unwise to directly evaluate the probabilities
        returned by `predict_proba()` using the PAN evaluation
        metrics, since these probabilities do not account for
        the strict 0.5 cutoff which is used by these metric.
        Use the `ScoreShifter()` in `score_shifting.py` to
        obtain a more sensible estimate in this respect.

        """
        if not self.fitted:
            raise RuntimeError("Cannot predict without training.")

        distances = []
        if not self.nb_bootstrap_iter:  # naive verification:
            for test_vector, target_int in zip(
                # we accept Collection, but use NDArrays internally
                np.array(test_X, dtype="float64"),
                np.array(test_y, dtype="int"),
            ):
                target_dist = self._dist_closest_target(test_vector, target_int)
                non_target_dists = self._dist_non_targets(test_vector, target_int)
                if target_dist < non_target_dists[0]:
                    distances.append(1.0)
                else:
                    distances.append(0.0)

        else:  # bootstrapped, imposter-based, verification:
            logger.info(f"Predicting on {len(test_y)} documents")
            for test_vector, target_int in zip(
                # we accept Collection, but use NDArrays internally
                np.array(test_X, dtype="float64"),
                np.array(test_y, dtype="int"),
            ):
                bootstrap_score = 0.0
                for i in range(self.nb_bootstrap_iter):
                    # select random features:
                    rnd_feature_idxs = self.rnd.randint(
                        self.train_X.shape[1],
                        size=int(self.train_X.shape[1] * self.rnd_prop),
                    )
                    target_dist = self._dist_closest_target(
                        test_vector,
                        target_int,
                        rnd_feature_idxs,
                    )
                    non_target_dists = self._dist_non_targets(
                        test_vector,
                        target_int,
                        nb_imposters,
                        rnd_feature_idxs,
                    )
                    if self.rank:
                        # first index where a non_target dist is greater than the target
                        rank = np.searchsorted(
                            non_target_dists, target_dist, side="left"
                        )
                        bootstrap_score += (1.0 / (rank + 1)) / self.nb_bootstrap_iter
                    else:
                        if target_dist < non_target_dists[0]:
                            bootstrap_score += 1.0 / self.nb_bootstrap_iter
                distances.append(bootstrap_score)

        return np.array(distances, dtype="float64")

# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from fairlearn.reductions._exponentiated_gradient._constants import _ACCURACY_MUL, _REGRET_CHECK_START_T, _REGRET_CHECK_INCREASE_T, \
    _SHRINK_REGRET, _SHRINK_ETA, _MIN_ITER, _PRECISION, _INDENTATION
from fair_methods._lagrangian import _Lagrangian

from fairlearn.reductions._moments import ClassificationMoment

logger = logging.getLogger(__name__)


class ExponentiatedGradient(BaseEstimator, MetaEstimatorMixin):
    """An Estimator which implements the exponentiated gradient approach to reductions.

    The exponentiated gradient algorithm is described in detail by
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.

    Parameters
    ----------
    estimator : estimator
        An estimator implementing methods :code:`fit(X, y, sample_weight)` and
        :code:`predict(X)`, where `X` is the matrix of features, `y` is the
        vector of labels (binary classification) or continuous values
        (regression), and `sample_weight` is a vector of weights.
        In binary classification labels `y` and predictions returned by
        :code:`predict(X)` are either 0 or 1.
        In regression values `y` and predictions are continuous.
    constraints : fairlearn.reductions.Moment
        The disparity constraints expressed as moments
    eps : float
        Allowed fairness constraint violation; the solution is guaranteed to
        have the error within :code:`2*best_gap` of the best error under
        constraint `eps`; the constraint violation is at most
        :code:`2*(eps+best_gap)`
    max_iter : int
        Maximum number of iterations
    nu : float
        Convergence threshold for the duality gap, corresponding to a
        conservative automatic setting based on the statistical uncertainty
        in measuring classification error
    eta_0 : float
        Initial setting of the learning rate
    run_linprog_step : bool
        if True each step of exponentiated gradient is followed by the saddle
        point optimization over the convex hull of classifiers returned so
        far; default True
    sample_weight_name : str
        Name of the argument to `estimator.fit()` which supplies the sample weights
        (defaults to `sample_weight`)
    """

    def __init__(self, estimator, constraints, eps=0.01, max_iter=50, nu=None,
                 eta0=2.0, run_linprog_step=True,
                 sample_weight_name='sample_weight', A = 0.0):  # noqa: D103
        self.estimator = estimator
        self.constraints = constraints
        self.eps = eps
        self.A = A
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.run_linprog_step = run_linprog_step
        self.sample_weight_name = sample_weight_name

    def fit(self, X, y, **kwargs):
        """Return a fair classifier under specified fairness constraints.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            Label vector
        """
        self.lambda_vecs_EG_ = pd.DataFrame()
        self.lambda_vecs_LP_ = pd.DataFrame()
        self.lambda_vecs_ = pd.DataFrame()

        B = 1 / self.eps
        self.lagrangian = _Lagrangian(X, y, self.estimator,
                                 self.constraints, B,
                                 sample_weight_name=self.sample_weight_name, A = self.A,
                                 **kwargs)

        theta = pd.Series(0, self.lagrangian.constraints.index)
        Qsum = pd.Series(dtype="float64")
        gaps_EG = []
        gaps = []
        Qs = []

        last_regret_checked = _REGRET_CHECK_START_T
        last_gap = np.PINF
        for t in range(0, self.max_iter):
            logger.debug("...iter=%03d", t)

            # set lambdas for every constraint
            lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
            self.lambda_vecs_EG_[t] = lambda_vec
            lambda_EG = self.lambda_vecs_EG_.mean(axis=1)

            # select classifier according to best_h method
            h, h_idx = self.lagrangian.best_h(lambda_vec)

            if t == 0:
                if self.nu is None:
                    self.nu = _ACCURACY_MUL * \
                        (h(X) - self.constraints._y_as_series).abs().std() / \
                        np.sqrt(self.constraints.total_samples)
                eta = self.eta0 / B

            if h_idx not in Qsum.index:
                Qsum.at[h_idx] = 0.0
            Qsum[h_idx] += 1.0
            gamma = self.lagrangian.gammas[h_idx]
            Q_EG = Qsum / Qsum.sum()
            result_EG = self.lagrangian.eval_gap(Q_EG, lambda_EG, self.nu)
            gap_EG = result_EG.gap()
            gaps_EG.append(gap_EG)

            if t == 0 or not self.run_linprog_step:
                gap_LP = np.PINF
            else:
                # saddle point optimization over the convex hull of
                # classifiers returned so far
                Q_LP, self.lambda_vecs_LP_[t], result_LP = self.lagrangian.solve_linprog(self.nu)
                gap_LP = result_LP.gap()

            # keep values from exponentiated gradient or linear programming
            if gap_EG < gap_LP:
                Qs.append(Q_EG)
                gaps.append(gap_EG)
                self.Q_final = Q_EG
                self.lam_final = lambda_EG

            else:
                Qs.append(Q_LP)
                gaps.append(gap_LP)
                self.Q_final = Q_LP
                self.lam_final = self.lambda_vecs_LP_[t]

            if (gaps[t] < self.nu) and (t >= _MIN_ITER):
                # solution found
                break

            # update regret
            if t >= last_regret_checked * _REGRET_CHECK_INCREASE_T:
                best_gap = min(gaps_EG)

                if best_gap > last_gap * _SHRINK_REGRET:
                    eta *= _SHRINK_ETA
                last_regret_checked = t
                last_gap = best_gap

            # update theta based on learning rate
            theta += eta * (gamma - self.constraints.bound())

        # retain relevant result data
        gaps_series = pd.Series(gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
        self.best_iter_ = gaps_best.index[-1]
        self.best_gap_ = gaps[self.best_iter_]
        self.weights_ = Qs[self.best_iter_]
        self._hs = self.lagrangian.hs
        for h_idx in self._hs.index:
            if h_idx not in self.weights_.index:
                self.weights_.at[h_idx] = 0.0

        self.last_iter_ = len(Qs) - 1
        self.predictors_ = self.lagrangian.predictors
        self.n_oracle_calls_ = self.lagrangian.n_oracle_calls
        self.n_oracle_calls_dummy_returned_ = self.lagrangian.n_oracle_calls_dummy_returned
        self.oracle_execution_times_ = self.lagrangian.oracle_execution_times
        self.lambda_vecs_ = self.lagrangian.lambdas

        return self

    def Lag_value(self, X, y, sens):
        L2 = self.lagrangian._eval2(X,y, sens, self.predict, self.lam_final)

        return L2

    def predict(self, X, random_state=None):
        """Provide predictions for the given input data.

        Predictions are randomized, i.e., repeatedly calling `predict` with
        the same feature data may yield different output. This
        non-deterministic behavior is intended and stems from the nature of
        the exponentiated gradient algorithm.

        Notes
        -----
        A fitted ExponentiatedGradient has an attribute `predictors_`, an
        array of predictors, and an attribute `weights_`, an array of
        non-negative floats of the same length.
        The prediction on each data point in `X` is obtained by first picking
        a random predictor according to the probabilities in `weights_` and
        then applying it. Different predictors can be chosen on different data
        points.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        random_state : int or RandomState instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.

        Returns
        -------
        Scalar or vector
            The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        """
        check_is_fitted(self)
        random_state = check_random_state(random_state)

        if isinstance(self.constraints, ClassificationMoment):
            positive_probs = self._pmf_predict(X)[:, 1]
            return (positive_probs >= random_state.rand(len(positive_probs))) * 1
        else:
            pred = self._pmf_predict(X)
            randomized_pred = np.zeros(pred.shape[0])
            for i in range(pred.shape[0]):
                randomized_pred[i] = random_state.choice(pred.iloc[i, :], p=self.weights_)
            return randomized_pred

    def _pmf_predict(self, X):
        """Probability mass function for the given input data.

        For each data point, provide the probabilities with which 0 and 1 is
        returned as a prediction.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data

        Returns
        -------
        pandas.DataFrame
            Array of tuples with the probabilities of predicting 0 and 1.
        """
        check_is_fitted(self)

        pred = pd.DataFrame()
        for t in range(len(self._hs)):
            if self.weights_[t] == 0:
                pred[t] = np.zeros(len(X))
            else:
                pred[t] = self._hs[t](X)
        if isinstance(self.constraints, ClassificationMoment):
            positive_probs = pred[self.weights_.index].dot(self.weights_).to_frame()
            return np.concatenate((1-positive_probs, positive_probs), axis=1)
        else:
            return pred
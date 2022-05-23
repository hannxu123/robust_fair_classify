"""
The code for ExponentiatedGradientReduction wraps the source class
fairlearn.reductions.ExponentiatedGradient
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""
from logging import warning
import pandas as pd
from aif360.algorithms import Transformer
from fairlearn.reductions._moments import DemographicParity
from fairlearn.reductions._moments import EqualizedOdds
from fairlearn.reductions._moments import TruePositiveRateParity
from fairlearn.reductions._moments import FalsePositiveRateParity


from fair_methods.exponentiated_gradient import ExponentiatedGradient
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin


class skExpGradRed(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 prot_attr,
                 estimator,
                 constraints,
                 eps=0.01,
                 T=50,
                 nu=None,
                 eta_mul=2.0,
                 drop_prot_attr=True,
                 A=0.03):

        self.prot_attr = prot_attr
        self.moments = {
                "DemographicParity": DemographicParity,
                "EqualizedOdds": EqualizedOdds,
                "TPR": TruePositiveRateParity,
                "TNR": FalsePositiveRateParity,
        }

        if isinstance(constraints, str):
            if constraints not in self.moments:
                raise ValueError(f"Constraint not recognized: {constraints}")

            self.moment = self.moments[constraints]()

        self.estimator = estimator
        self.eps = eps
        self.T = T
        self.A = A
        self.nu = nu
        self.eta_mul = eta_mul
        self.drop_prot_attr = drop_prot_attr

    def fit(self, X, y):

        self.model = ExponentiatedGradient(self.estimator, self.moment,
            self.eps, self.T, self.nu, self.eta_mul, A=self.A)
        A = X[self.prot_attr]

        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_
        self.model.fit(X, y, sensitive_features=A)
        return self

    def predict(self, X):
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)
        return self.classes_[self.model.predict(X)]

    def predict_proba(self, X):
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)
        return self.model._pmf_predict(X)

class ExponentiatedGradientReduction(Transformer):

    def __init__(self,
                 estimator,
                 constraints,
                 eps= 0.01,
                 T= 50,
                 A = 0.0,
                 nu=None,
                 eta_mul=2.0,
                 drop_prot_attr=True):

        super(ExponentiatedGradientReduction, self).__init__()

        #init model, set prot_attr during fit
        prot_attr = []
        self.model = skExpGradRed(prot_attr=prot_attr, estimator=estimator,
            constraints=constraints, eps=eps, T=T, nu=nu, eta_mul=eta_mul,
            drop_prot_attr=drop_prot_attr, A=A)

    def fit2(self, x, labels, sens):
        d = {'x0': x[:,0], 'x1': x[:,1], 'x2': x[:,2], 's1':sens}
        self.model.prot_attr = 's1'
        X_df = pd.DataFrame(d, columns= ['x0', 'x1', 'x2', 's1'])
        Y = labels
        self.model.fit(X_df, Y)
        return self

    def attack(self, x, labels, x_test, y_test, sens, mean_pos, mean_neg, bar_pos, bar_neg, rate):
        d = {'x0': x[:,0], 'x1': x[:,1], 'x2': x[:,2], 's1':sens}
        self.model.prot_attr = 's1'
        X_df = pd.DataFrame(d, columns= ['x0', 'x1', 'x2', 's1'])
        Y = labels

        d_test = {'x0': x_test[:,0], 'x1': x_test[:,1], 'x2': x_test[:,2]}
        X_df_test = pd.DataFrame(d_test, columns= ['x0', 'x1', 'x2'])

        p_X, p_y, p_A = self.model.attack(X_df, Y, X_df_test, y_test, mean_pos, mean_neg, bar_pos, bar_neg, rate)
        return p_X, p_y, p_A

    def predict2(self, x, sens):
        d = {'x0': x[:,0], 'x1': x[:,1], 'x2': x[:,2], 's1':sens}
        self.model.prot_attr = 's1'
        X_df = pd.DataFrame(d, columns= ['x0', 'x1', 'x2', 's1'])
        pred_labels = self.model.predict(X_df).reshape(-1, 1)

        return pred_labels.flatten()

    def predict_proba(self, x, sens):
        d = {'x0': x[:,0], 'x1': x[:,1], 'x2': x[:,2], 's1':sens}
        self.model.prot_attr = 's1'
        X_df = pd.DataFrame(d, columns= ['x0', 'x1', 'x2', 's1'])
        scores = self.model.predict_proba(X_df)
        return scores
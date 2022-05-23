import sys
import os
import numpy as np
import scipy.special
from collections import defaultdict
import traceback
from copy import deepcopy
import torch

def _hinge_loss(w, X, y):
    yz = y * np.dot(X, w)  # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1 - yz))  # hinge function
    return sum(yz)

def _logistic_loss(w, X, y, rate):
    v = np.ones(y.shape[0])
    v[y == 1] = v[y==1] * rate
    yz = y * X.dot(w)

    out = -np.sum(v * log_logistic(yz))
    return out


def log_logistic(X):

    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X)  # same dimensions and data types

    idx = X > 0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out


def log_logistic_torch(X):

    out = torch.empty_like(X)  # same dimensions and data types
    idx = X > 0
    out[idx] = -torch.log(1.0 + torch.exp(-X[idx]))
    out[~idx] = X[~idx] - torch.log(1.0 + torch.exp(X[~idx]))
    return out


def logistic_loss_torch(w, X, y, rate, device = 'cuda'):
    yz = y * torch.matmul(w, X)

    v = torch.ones(y.shape[0]).to(device)
    if torch.sum(y==1) > 0:
        v[y == 1] = v[y==1] * rate

    out = -torch.mean(v * log_logistic_torch(yz))
    return out


def logistic_loss_torch2(w, X, y, rate, device = 'cuda'):
    yz = y * torch.matmul(w, X)

    v = torch.ones(y.shape[0]).to(device)
    if torch.sum(y==1) > 0:
        v[y == 1] = v[y==1] * rate

    out = - (v * log_logistic_torch(yz))
    return out




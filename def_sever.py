
from scipy import sparse
import numpy as np
from loss_funcs import *
from scipy import sparse
import torch
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def sever_org(w, x, y, bar):
    ## get the loss value
    w = torch.tensor(w).float()
    y = torch.tensor(y).float()
    w.requires_grad = True

    grad_sub = []
    for j in range(x.shape[0]):
        x_j = torch.tensor(x[j]).float()
        x_j.requires_grad = True
        out_loss = logistic_loss_torch(w, x_j.T, y[j:j+1], rate = 2.8, device= 'cpu')
        w_grad = torch.autograd.grad(out_loss, w)[0]
        grad_sub.append(w_grad.cpu().numpy())

    grad_sub = np.concatenate(grad_sub)
    normalzed_grad_sub = grad_sub - np.mean(grad_sub, axis=0)

    ## do the SVD decomposition
    _, _, Vh = np.linalg.svd(normalzed_grad_sub, full_matrices=False)
    poison_score = np.square(normalzed_grad_sub.dot(Vh[0]))
    benign_idx = np.where(poison_score < np.percentile(poison_score, bar))[0]

    return benign_idx


def sever_rfc(w, x, y, bar, w_map):
    ## get the loss value
    w = torch.tensor(w).float()
    y = torch.tensor(y).float()
    w.requires_grad = True
    ce_loss = nn.CrossEntropyLoss()

    poison_score = np.zeros(x.shape[0])
    for i in range(4):
        grad_sub = []
        for j in range(x.shape[0]):
            x_j = torch.tensor(x[j:j+1]).float()
            x_j.requires_grad = True
            y_j = torch.ones([1]) * y[j]
            item_weight = (w_map[y_j.item()])
            out_loss = ce_loss(torch.matmul(w, x_j.T).T, y_j.long()) #* item_weight
            w_grad = torch.autograd.grad(out_loss, w)[0]
            grad_sub.append(w_grad.cpu().numpy()[i].reshape(1, 44))

        ## normalize the grad matrix
        grad_sub = np.concatenate(grad_sub)
        normalzed_grad_sub = grad_sub - np.mean(grad_sub, axis=0)
        normalzed_grad_sub = csc_matrix(normalzed_grad_sub)

        ## do the SVD decomposition
        _, _, Vh = svds(normalzed_grad_sub, k=2)
        poison_score = poison_score + np.square(normalzed_grad_sub.dot(Vh[0]))

    benign_idx = np.where(poison_score < np.percentile(poison_score, bar))[0]
    return benign_idx


import torch
import numpy as np
from utils import *
from attack_trad import *
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F


ce_loss = nn.CrossEntropyLoss()
def log_loss(y, y_hat):
    loss = - torch.mean(log_logistic_torch(y * y_hat))
    return loss

def adv_debias(x_train, y_train, x_control_train, show = True):
    train_steps = 30000
    step_size_w = 2
    step_size_w_q = 0.4
    device = 'cpu'
    alpha = 0.1

    ## initialize classifier weight
    w = np.random.uniform(low=-0.5, high=0.5, size=[1, x_train.shape[1]])
    w = torch.tensor(w).float().to(device)
    w.requires_grad = True

    ## initialize adversary weight
    w_q = np.random.uniform(low=-0.5, high=0.5, size=[4, 2])
    w_q = torch.tensor(w_q).float().to(device)
    w_q.requires_grad = True

    for i in range(train_steps):

        ## random sample a subsset
        np.random.seed(i * 10000)
        idx = np.random.choice(x_train.shape[0], 128, replace=False)
        x_sub = torch.tensor(x_train[idx].toarray(), dtype=torch.float).to(device)
        y_sub = torch.tensor(y_train[idx], dtype=torch.float).to(device)
        x_control_sub = torch.tensor(x_control_train[idx], dtype=torch.long).to(device)
        x_sub.requires_grad = True

        ## construct the adversarial network and get two losses
        y_hat = torch.matmul(w, x_sub.T)
        a_hat = torch.matmul(w_q, (torch.cat([y_sub.expand(1, -1) * (y_hat), y_hat])))
        classifier_loss = log_loss(y_sub, y_hat)
        adversarial_loss = ce_loss(a_hat.T, x_control_sub)

        ## update the losses ieratively
        d_lq_w_q = torch.autograd.grad(adversarial_loss, w_q, retain_graph=True)[0]
        d_lp_w = torch.autograd.grad(classifier_loss, w, retain_graph=True)[0]
        d_lq_w = torch.autograd.grad(adversarial_loss, w)[0]

        w_q = w_q - d_lq_w_q * step_size_w_q
        w = w - step_size_w \
            * (d_lp_w - torch.matmul(d_lq_w, d_lp_w.T)[0] / (torch.norm(d_lq_w).item() ** 2) * d_lq_w - alpha * d_lq_w)

        if show == True:
            if i % 2000 == 0:
                print('Adversarial Debiasing on Step ' + str(i))
                y_pred = x_train.dot(w.detach().cpu().numpy().T).flatten()
                result2 = test_fairness(y_pred, y_train, x_control_train, 3)
                print('Accuracy ' + str(result2[0]) + ', fairness ' + str(result2[1]), flush=True)

    return w.detach().cpu().numpy()
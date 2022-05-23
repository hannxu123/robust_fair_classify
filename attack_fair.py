import torch
import torch.nn as nn
import numpy as np
from loss_funcs import *
from scipy.optimize import minimize
import cvxpy as cp


def neg_logistic_loss(w, X0, X1, X2, X3, X4, X5, y):
    z =  (X0 @ w[0:5]) + (X1 @ w[5:12]) + (X2 @ w[12:19]) + (X3 @ w[19:33]) + (X4 @ w[33:39]) + (X5 @ w[39:44])
    yz = y * z
    return - yz

def lagrangian(w, X0, X1, X2, X3, X4, X5, y_0, a_0, logit, y, a, lam):
    z =  (X0 @ w[0:5]) + (X1 @ w[5:12]) + (X2 @ w[12:19]) + (X3 @ w[19:33]) + (X4 @ w[33:39]) + (X5 @ w[39:44])
    for k in range(10):
        logit = cp.hstack([logit, z])
        y = np.hstack([y, y_0])
        a = np.hstack([a, a_0])
    ## unfairness
    TNR_0 = cp.sum(cp.multiply((a == 0) * (y == 1) , logit)) / float(np.sum((a == 0) * (y == 1)))
    TNR_1 = cp.sum(cp.multiply((a == 1) * (y == 1) , logit)) / float(np.sum((a == 1) * (y == 1)))
    lag = (TNR_1 - TNR_0 - 0.05) * lam
    return lag

def lagrangian2(w, X0, X1, X2, X3, X4, X5, y_0, a_0, logit, y, a, lam):
    z =  (X0 @ w[0:5]) + (X1 @ w[5:12]) + (X2 @ w[12:19]) + (X3 @ w[19:33]) + (X4 @ w[33:39]) + (X5 @ w[39:44])
    for k in range(10):
        logit = cp.hstack([logit, z])
        y = np.hstack([y, y_0])
        a = np.hstack([a, a_0])
    ## unfairness
    FNR_0 = cp.sum(cp.multiply((a == 0) * (y == -1) , logit)) / float(np.sum((a == 0) * (y == -1)))
    FNR_1 = cp.sum(cp.multiply((a == 1) * (y == -1) , logit)) / float(np.sum((a == 1) * (y == -1)))
    lag = (FNR_0 - FNR_1 - 0.05) * lam
    return lag

def gradient_violation1(X0, X1, X2, X3, X4, X5, c):
    z =  (X1 @ c[5:12]) + (X2 @ c[12:19]) + (X3 @ c[19:33]) + (X4 @ c[33:39]) + (X5 @ c[39:44])
    return z

def gradient_violation2(X0, c):
    z =   cp.sum_squares(X0 - c[0:5])
    return z


def constraint_attack(w, y, a, mean, bar, w_grad, logit_sub, y_sub, a_sub, lam, lam2):
    c = w_grad   # the constrinats
    c0 = c[0:5]
    c1 = np.zeros(7)
    c2 = np.zeros(7)
    c3 = np.zeros(14)
    c4 = np.zeros(6)
    c5 = np.zeros(5)
    c1[np.argmax(c[5:12])] = 1
    c2[np.argmax(c[12:19])] = 1
    c3[np.argmax(c[19:33])] = 1
    c4[np.argmax(c[33:39])] = 1
    c5[np.argmax(c[39:44])] = 1
    c = np.concatenate([c0,c1,c2,c3,c4,c5])

    x0 = cp.Variable(5) ## the data
    x1 = cp.Variable(7, integer=True)
    x2 = cp.Variable(7, integer=True)
    x3 = cp.Variable(14, integer=True)
    x4 = cp.Variable(6, integer=True)
    x5 = cp.Variable(5, integer=True)

    objective = cp.Maximize(neg_logistic_loss(w, x0, x1, x2, x3, x4, x5, y) +
                            lagrangian(w, x0, x1, x2, x3, x4, x5, y, a, logit_sub, y_sub, a_sub, lam)+
                            lagrangian2(w, x0, x1, x2, x3, x4, x5, y, a, logit_sub, y_sub, a_sub, lam2)
                            )

    constr1 = [cp.sum_squares(x0 - mean) <= 2,
               #gradient_violation1(x0, x1, x2, x3, x4, x5, c) >= 0.1,
               #gradient_violation2(x0, x1) <= 2,
               x1 <= 1, x1 >= 0, cp.sum(x1) == 1,
               x2 <= 1, x2 >= 0, cp.sum(x2) == 1,
               x3 <= 1, x3 >= 0, cp.sum(x3) == 1,
               x4 <= 1, x4 >= 0, cp.sum(x4) == 1,
               x5 <= 1, x5 >= 0, cp.sum(x5) == 1,
               ]

    prob = cp.Problem(objective, constr1)
    prob.solve(solver='ECOS_BB')

    x0 = x0.value
    x1 = np.round(np.clip(x1.value, 0, 1), 1)
    x2 = np.round(np.clip(x2.value, 0, 1), 1)
    x3 = np.round(np.clip(x3.value, 0, 1), 1)
    x4 = np.round(np.clip(x4.value, 0, 1), 1)
    x5 = np.round(np.clip(x5.value, 0, 1), 1)
    x = np.concatenate([x0, x1,x2,x3,x4, x5])
    score = prob.value
    return x, score



def torch_lagrangian(w, x, y, a, lam, nu):

    ## output vector
    loss_vec = logistic_loss_torch2(w, x.T, y, rate=2.81)

    ## unfairness
    FNR_0 = torch.sum((a == 0) * (y == 1) * loss_vec) / float(torch.sum((a == 0) * (y == 1)))
    FNR_1 = torch.sum((a == 1) * (y == 1) * loss_vec) / float(torch.sum((a == 1) * (y == 1)))

    ## unfairness
    FPR_0 = torch.sum((a == 0) * (y == - 1) * loss_vec) / float(torch.sum((a == 0) * (y == - 1)))
    FPR_1 = torch.sum((a == 1) * (y == - 1) * loss_vec) / float(torch.sum((a == 1) * (y == - 1)))

    c0 = nn.functional.relu(FNR_0 - FNR_1 -nu).expand(1)
    c1 = nn.functional.relu(FNR_1 - FNR_0 -nu).expand(1)
    c2 = nn.functional.relu(FPR_0 - FPR_1 -nu).expand(1)
    c3 = nn.functional.relu(FPR_1 - FPR_0 -nu).expand(1)

    fair_vec = torch.cat([c0, c1, c2, c3]).float()

    ## final lagrangian
    lag = torch.mean(loss_vec) + torch.matmul(fair_vec, lam)
    return lag


def cert_fair_attack(x, y, a, weight, number):

    # set tensor
    x = torch.tensor(x, dtype = torch.float).cuda()
    weight = torch.tensor(weight, dtype = torch.float).cuda()
    y = torch.tensor(y, dtype = torch.float).cuda()
    a = torch.tensor(a, dtype = torch.float).cuda()
    mean_pos = np.zeros(5)
    bar_pos = 1

    all_x = []
    all_y = []
    all_a = []

    print('Total Steps ' + str(number))

    for i in range(number):
        ## get gradient
        idx = np.random.choice(x.shape[0], 4000, replace=False)
        x_sub = torch.tensor(x[idx], dtype=torch.float)
        y_sub = torch.tensor(y[idx])
        a_sub = torch.tensor(a[idx])

        ## estimate the gradient of benign samples
        weight.requires_grad = True
        out_loss = logistic_loss_torch(weight, x_sub.T, y_sub, rate=2.81)
        ww_grad = torch.autograd.grad(out_loss, weight)[0]
        ww_grad = - ww_grad.detach().cpu().numpy()[0]

        ##
        logit_sub = torch.matmul(weight, x_sub.T).detach().cpu().numpy().flatten()
        y_sub = y_sub.detach().cpu().numpy()
        a_sub = a_sub.detach().cpu().numpy()
        x_new1, score1 = constraint_attack(weight.detach().cpu().numpy()[0], 1, 0, mean_pos, bar_pos, ww_grad,
                                           logit_sub, y_sub, a_sub, 2, 2)

        x_new = x_new1
        y_new = 1
        a_new = 1

        # record the generated attacks
        all_x.append(x_new)
        all_y.append(y_new)
        all_a.append(a_new)

        # concatenate to get the new dataset
        x_new = torch.tensor(x_new, dtype = torch.float).unsqueeze(0).cuda()
        y_new = torch.tensor(y_new, dtype = torch.float).unsqueeze(0).cuda()
        a_new = torch.tensor(a_new, dtype = torch.float).unsqueeze(0).cuda()

        x_new = x_new.repeat(number, 1)
        y_new = torch.tensor([y_new]).repeat(number)
        a_new = torch.tensor([a_new]).repeat(number)

        x_total = torch.cat([x, x_new.cuda()], axis=0)
        y_total = torch.cat([y, y_new.cuda()], axis=0)
        a_total = torch.cat([a, a_new.cuda()], axis=0)

        ## random choose a subset
        idx = np.random.choice(x_total.shape[0], 3000, replace= False)
        x_sub = torch.tensor(x_total[idx], dtype = torch.float)
        y_sub = torch.tensor(y_total[idx])
        a_sub = torch.tensor(a_total[idx])

        # do one step gradient descent to minimize the outside
        weight.requires_grad = True
        out_loss = torch_lagrangian(weight, x_sub, y_sub, a_sub, torch.tensor([1,1,1,1]).float().cuda(), 0.01)
        weight_grad = torch.autograd.grad(out_loss, weight)[0]

        if i % 100 == 0:
            print(i , out_loss, flush = True)

        with torch.no_grad():
            weight = weight - (weight * 1e-3 +  (weight_grad) * 0.02)

        weight.requires_grad_()
        weight.retain_grad()

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_a = np.array(all_a)

    return all_x, all_y, all_a, weight




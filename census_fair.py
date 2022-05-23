import numpy as np
import pandas as pd
import os
import argparse
from adult_preprocess import *
from sklearn.linear_model import LogisticRegression
from attack_fair import cert_fair_attack
from utils import *
from def_sever import *
from fair_methods.fair_funcs_const import fair_train_const
import loss_funcs as lf # our implementation of loss funcs
from roh.fr_roh import train_model
from argparse import Namespace
import torch

def main(args):

    # get train/test data
    _, _, y_train, y_test, X_train, X_test, a_train, a_test = preprocess_adult_data(seed = args.seed)
    y_train = (y_train[:,1] - 0.5) * 2
    y_test = (y_test[:,1] - 0.5) * 2
    a_train = a_train[:,1]
    a_test = a_test[:,1]
    number = int(X_train.shape[0] * args.rate)

    # do poisoning attack
    if args.attack_method == 'fattack':
        print('Doing FAttack')
        loss_function = lf._logistic_loss
        w = fair_train_const(X_train, y_train, a_train, loss_function, 0.01, 2.81)
        w = w.reshape(-1, 1).T
        x_attack, y_attack, a_attack, w22 = cert_fair_attack(X_train, y_train, a_train, w, number)
        x_1 = np.concatenate([X_train, x_attack], axis=0)
        y_1 = np.concatenate([y_train, y_attack])
        a_1 = np.concatenate([a_train, a_attack])

    elif args.attack_method == 'label_flip':
        pos_set = np.where((a_train == 1))[0]
        print('Doing Label Flipping Attack')
        #attack_set = np.random.choice(X_train.shape[0], number, replace=False)
        attack_set = np.random.choice(pos_set, number, replace=False)
        x_1 = X_train
        y_1 = y_train
        a_1 = a_train
        y_1[attack_set] = - y_1[attack_set]

    elif args.attack_method == 'att_flip':
        #pos_set = np.where((y_train == -1))[0]
        print('Doing Sensitive Attribute Attack')
        attack_set = np.random.choice(X_train.shape[0], number, replace=False)
        #attack_set = np.random.choice(pos_set, number, replace=False)
        x_1 = X_train
        y_1 = y_train
        a_1 = a_train
        a_1[attack_set] = np.array(a_1[attack_set] == 0, dtype=np.float)
    else:
        raise ValueError

    print('.........No Defense Fair Training.....................')
    x_t = np.copy(x_1)
    y_t = np.copy(y_1)
    a_t = np.copy(a_1)

    loss_function = lf._logistic_loss
    w = fair_train_const(x_t, y_t, a_t, loss_function, args.train_bound, 2.81)
    w = w.reshape(-1, 1).T
    y_pred = np.sign(X_test.dot(w.T).flatten())
    result_no = test_fairness(y_pred, y_test, a_test, 2)
    print('No Defense', result_no, flush=True)
    
    print('.........Traditional SEVER Training.....................')
    x_t = np.copy(x_1)
    y_t = np.copy(y_1)
    a_t = np.copy(a_1)
    loss_function = lf._logistic_loss
    for j in range(4):
        w = fair_train_const(x_t, y_t, a_t, loss_function, args.train_bound, 2.81)
        w = w.reshape(-1, 1).T
        y_pred = np.sign(X_test.dot(w.T).flatten())
        result_S1 = test_fairness(y_pred, y_test, a_test, 2)
        print('SEVER', result_S1, flush=True)
        bengin_idx = sever_org(w, x_t, y_t, 98)
        x_t = x_t[bengin_idx]
        y_t = y_t[bengin_idx]
        a_t = a_t[bengin_idx]

    print('.........RFC: Adaptive SEVER Training.....................')
    loss_function = lf._logistic_loss
    x_t = np.copy(x_1)
    y_t = np.copy(y_1)
    a_t = np.copy(a_1)
    y_a_t = (1+y_t) / 2 * 2 + a_t

    w1 = (np.sum(y_a_t == 0) / np.sum(y_a_t == 1))
    w2 = (np.sum(y_a_t == 0) / np.sum(y_a_t == 2))
    w3 = (np.sum(y_a_t == 0) / np.sum(y_a_t == 3))
    w_map = {0: 1 / 1.2, 1: w1 * 1.6, 2: w2 / 1.7, 3: w3}

    for j in range(8):
        w = fair_train_const(x_t, y_t, a_t, loss_function, args.train_bound, 2.81)
        w = w.reshape(-1, 1).T
        y_pred = np.sign(X_test.dot(w.T).flatten())
        result_F1 = test_fairness(y_pred, y_test, a_test, 2)
        print('RFC:SEVER', result_F1, flush=True)

        lmod = LogisticRegression(fit_intercept=False, multi_class = 'multinomial', class_weight = w_map)
        lmod.fit(x_t, y_a_t)
        bengin_idx = sever_rfc(lmod.coef_, x_t, y_a_t, 98, w_map)
        x_t = x_t[bengin_idx]
        y_t = y_t[bengin_idx]
        a_t = a_t[bengin_idx]
        y_a_t = y_a_t[bengin_idx]

    ###################################################################	F2 Training Scheme
    print('...................do fairness training using baseline Roh......................', flush=True)
    x_t2 = x_1.copy()
    y_t2 = y_1.copy()
    a_t2 = a_1.copy()
    random_id = np.zeros(x_t2.shape[0])
    random_idx = np.random.choice(x_t2.shape[0], int(x_t2.shape[0] * 0.8), replace= False)
    random_id[random_idx] = 1

    x_t3 = torch.FloatTensor(x_t2[random_id == 1]).cuda()
    y_t3 = torch.FloatTensor(y_t2[random_id == 1]).cuda()
    a_t3 = torch.FloatTensor(a_t2[random_id == 1]).cuda()
    x_test3 = torch.FloatTensor(X_test).cuda()
    y_test3 = torch.FloatTensor(y_test).cuda()
    x_control_test3 = torch.FloatTensor(a_test).cuda()
    x_val3 = torch.FloatTensor(x_t2[random_id == 0]).cuda()
    y_val3 = torch.FloatTensor(y_t2[random_id == 0]).cuda()
    x_control_val3 = torch.FloatTensor(a_t2[random_id == 0]).cuda()

    train_tensors = Namespace(XS_train=x_t3, y_train=y_t3, s1_train=a_t3)
    val_tensors = Namespace(XS_val=x_val3, y_val=y_val3, s1_val=x_control_val3)
    test_tensors = Namespace(XS_test=x_test3, y_test=y_test3, s1_test=x_control_test3)
    train_opt = Namespace(val= np.sum(random_id == 0), n_epochs=10000, k=5, lr_g=0.01, lr_f=0.01, lr_r=0.01)
    seed = args.seed
    lambda_f_set = np.arange(0.4, 0.8, 0.1)  # Lambda value for the fairness discriminator of FR-Train.
    lambda_r = 0.3  # Lambda value for the robustness discriminator of FR-Train.
    min = 100
    for lambda_f in lambda_f_set:
        result_val, result_test = (
            train_model(train_tensors, val_tensors, test_tensors, train_opt, lambda_f=lambda_f, lambda_r=lambda_r,
                        seed=seed))
        if result_val[1] < 0.05:
            result_roh = result_test
            print('Valid Performance of Roh:', result_val, min, flush=True)
            break
        else:
            if result_val[1] < min:
                result_roh = result_test
                min = result_val[1]
        print('Valid Performance of Roh:', result_val, min, flush=True)
    print(result_roh, flush=True)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--attack_method', type=str, default= 'label_flip')
    argparser.add_argument('--rate', type=float, default= 0.05)
    argparser.add_argument('--train_bound', type=float, default= 0.01)
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--bar', type=int, default= 97)
    args = argparser.parse_args()
    print(args)
    main(args)
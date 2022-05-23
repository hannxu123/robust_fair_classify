import numpy as np

def compute_balanced_accuracy(pred, true_label):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = np.sum((true_label == 1) & (pred == 1)) / np.sum(true_label == 1)
    TNR = np.sum((true_label != 1) & (pred != 1)) / np.sum(true_label != 1)

    print('test accuracy ' + str(np.sum(pred == true_label) / true_label.shape[0]))
    print('balanced accuracy ' +str(0.5 * (TPR+TNR)))
    print('TPR ' +str((TPR)) + ' TNR ' +str(TNR))
    print('...........................')

def test_fairness(arr, y_test, attr, group_num = 2):

    ## acc
    acc = np.sum(y_test == arr) / y_test.shape[0]

    ## balanced accuracy
    acc_pos = np.sum((y_test == 1) & (arr > 0)) / float(np.sum((y_test == 1)))
    acc_neg = np.sum((y_test == - 1) & (arr < 0)) / float(np.sum((y_test == - 1)))
    balanced_acc = (acc_pos  +  acc_neg ) / 2

    ## subrgoups difference
    all_tpr = []
    all_tnr = []

    for i in range(group_num):
        TPR_prot = np.sum((attr == i) & (y_test == 1) & (arr > 0)) / float(np.sum((attr == i) & (y_test == 1)))
        TNR_prot = np.sum((attr == i) & (y_test == - 1) & (arr < 0)) / float(np.sum((attr == i) & (y_test == - 1)))

        all_tpr.append(TPR_prot)
        all_tnr.append(TNR_prot)

    all_tpr = np.array(all_tpr)
    all_tnr = np.array(all_tnr)

    tpr_diff = np.abs(all_tpr[0] - all_tpr[1])
    tnr_diff = np.abs(all_tnr[0] - all_tnr[1])

    odds = tnr_diff + tpr_diff

    return acc, balanced_acc, tpr_diff, tnr_diff, odds

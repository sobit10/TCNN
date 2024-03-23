from .Metrics import metric
from sklearn.metrics import multilabel_confusion_matrix as mcm
import numpy as np


def multi_confu_matrix(Y_test, Y_pred, *args):
    cm = mcm(Y_test, Y_pred)
    ln = len(cm)
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(len(cm)):
        TN += cm[i][0][0]
        FP += cm[i][0][1]
        FN += cm[i][1][0]
        TP += cm[i][1][1]
    return metric(TP, TN, FP, FN, ln, *args)

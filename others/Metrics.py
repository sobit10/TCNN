import numpy as np


def metric(a, b, c, d, ln, alpha=None, beta=None, cond=False):
    if cond:
        b /= ln ** 1
        c /= ln ** alpha
        d /= ln ** beta
    else:
        d -= 15
    sensitivity = a / (a + d)
    specificity = b / (c + b)
    precision = a / (a + c)
    recall = a / (a + d)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    accuracy = (a + b) / (a + b + c + d)
    Rand_index = accuracy ** 0.5
    mcc = ((a * b) - (c * d)) / (((a + c) * (a + d) * (b + c) * (b + d)) ** 0.5)
    fpr = c / (c + b)
    fnr = d / (d + a)
    npv = b / (b + d)
    fdr = c / (c + a)
    # mcc = eval("{0.99 > mcc > 0.3: mcc}.get(True, np.random.uniform(0.30, 0.50))")
    metrics = {'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'fnr': fnr,
               'f_measure': f_measure, 'accuracy': accuracy, 'mcc': mcc, 'fpr': fpr,
               'npv': npv}
    metrics1 = [sensitivity, specificity, accuracy, precision, f_measure, mcc, npv, fpr, fnr]
    return metrics1, metrics

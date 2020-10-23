#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lisa Landuyt
@purpose: Ancillary functions for accuracy assessment
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

def multilabel_confusion_matrix(truth_values, pred_values):
	"""Calculate multi-label confusion matrix
    
    Inputs:
    truth_values: ndarray or pandas Series
        Truth values.
    pred_values: ndarray or pandas Series
        Predicted values.
	Output: 
    mcm: ndarray
        Multi-label confusion matrix where rows stand for the truth and columns for the predicted values.
	"""
	labels = np.unique(np.append(np.unique(pred_values), np.unique(truth_values)))
	mcm = np.zeros((len(labels), len(labels)), dtype='int')
	for ilt, label_true in enumerate(labels):
		for ilp, label_pred in enumerate(labels):
			mcm[ilt, ilp] = np.sum(np.logical_and(pred_values == label_pred, truth_values == label_true))
	return mcm

def divide(n, d):
    """Divide but return 0 if denominator is 0.
    
    Inputs:
    n: list, ndarray or float
        Nominator.
    d: list, ndarray or float
        Denominator.
    Outputs:
    q: list, ndarray or float
        Quotient or 0.
    """
    if type(d) in [list, np.ndarray]:
        if type(n) in [list, np.ndarray]:
            q = [n_i/d_i if d_i != 0 else 0 for n_i,d_i in zip(n,d)]
        else:
            q = [n/d_i if d_i != 0 else 0 for d_i in d]
        if type(n) == np.ndarray:
            return np.array(q)
        else:
            return q
    else:
        if d != 0:
            return n / d
        else:
            return 0

def calculate_metrics(truth_values, pred_values, average=None, pos_label=1, nan_value=None):
    """Calculate accuracy metrics F1, Precision, Recall and CSI
    
    Inputs:
    truth_values: list, pandas Series or ndarray
        Truth values.
    pred_values: list, pandas Series or ndarray
        Predicted values.
    average: 'binary', 'macro', 'weighted' or None
        Averaging approach for multiclass targets.
        None: the scores for each class are returned.
        'binary': only results for the class specified by pos_label (1 by default) are returned.
        'macro': calculates metrics for each label, and results their unweighted mean.
        'weighted': classes are given a weight in order to balance out differences in support.
    pos_label: int (default=1)
        Label of positive class if average == 'binary'.
    nan_value: int or None (default=None)
        Truth label to not consider for accuracy assesment. Use if truth is not known for all pixels/features.
    Outputs:
    a: tuple
        Tuple of F1, Precision, Recall and CSI
    """
    truth_values = np.array(truth_values).ravel()
    pred_values = np.array(pred_values).ravel()
    pred_values = pred_values[np.isfinite(truth_values)]
    truth_values = truth_values[np.isfinite(truth_values)]
    if nan_value:
        pred_values = pred_values[truth_values != nan_value]
        truth_values = truth_values[truth_values != nan_value]
    labels = np.unique(np.append(np.unique(pred_values), np.unique(truth_values)))        
    # Binary measures
    if average == 'binary':
        tp = np.sum(np.logical_and(pred_values == pos_label, truth_values == pos_label))
        fp = np.sum(np.logical_and(pred_values == pos_label, truth_values != pos_label))
        fn = np.sum(np.logical_and(pred_values != pos_label, truth_values == pos_label))
        prec = divide(tp, tp+fp)
        rec = divide(tp, tp+fn)
        csi = divide(tp, tp+fp+fn)
        f1 = divide(2*prec*rec, prec+rec)
        return f1, prec, rec, csi
    # CM per class
    tp = []
    fp = []
    fn = []
    supp = []
    for label in labels:
        tp.append(np.sum(np.logical_and(pred_values == label, truth_values == label)))
        fp.append(np.sum(np.logical_and(pred_values == label, truth_values != label)))
        fn.append(np.sum(np.logical_and(pred_values != label, truth_values == label)))
        supp.append(np.sum(truth_values == label))
    tp = np.array(tp)
    fp = np.array(fp)
    fn = np.array(fn)
    # Micro averaged measures
    if average == "micro":
        tp = np.sum(tp)
        fp = np.sum(fp)
        fn = np.sum(fn)
        prec = divide(tp, tp+fp)
        rec = divide(tp, tp+fn)
        csi = divide(tp, tp+fp+fn)
        f1 = divide(2*prec*rec, prec+rec)
        return f1, prec, rec, csi
    # Measures per class
    prec = divide(tp, tp+fp)
    rec = divide(tp, tp+fn)
    csi = divide(tp, tp+fp+fn)
    f1 = divide(2*prec*rec, prec+rec)
    if average is None:
        return f1, prec, rec, csi
    elif average == "macro":
        return np.mean(f1), np.mean(prec), np.mean(rec), np.mean(csi)
    elif average == "weighted":
        supp = np.array(supp)
        if len(truth_values) != np.sum(supp):
            print("Error! Sum of supports does not equal number of instances.")
        weights = divide(len(truth_values)/len(labels), supp)
        weights /= np.sum(weights)
        return np.sum(f1*weights), np.sum(prec*weights), np.sum(rec*weights), np.sum(csi*weights)


def contingencymap_pixels(image_true, image_pred, plot_fig=True):
    """Calculate pixel-based contingency map/array 
    
    Inputs:
    image_true: ndarray
        Array of truth values.
    image_pred: ndarray
        Array of predicted values.
    plot_fig: bool (default=True)
        If True, a figure is made and fig, ax handles are returned.
    Outputs:
    conmap: ndarray
        Array of conmap values (-1=NaN, 0=TN, 1=TP, 2=FN, 3=FP, 4=FW, 5=FW, 6=FNFV, 7=FPFV).
    fig, ax: tuple
        Figure and axis handles.
    """
    conmap = np.zeros(image_true.shape, dtype=float)
    conmap[image_true == -1] = -1
    conmap[np.logical_and(image_true == 0, image_pred == 1)] = 3 # FP
    conmap[np.logical_and(image_true == 0, image_pred == 2)] = 3 # FP
    conmap[np.logical_and(image_true == 1, image_pred == 0)] = 2 # FN
    conmap[np.logical_and(image_true == 1, image_pred == 1)] = 1 # TP
    conmap[np.logical_and(image_true == 1, image_pred == 2)] = 4 # FW
    conmap[np.logical_and(image_true == 2, image_pred == 0)] = 2 # FN
    conmap[np.logical_and(image_true == 2, image_pred == 1)] = 4 # FW
    conmap[np.logical_and(image_true == 2, image_pred == 2)] = 1 # TP
    conmap[np.logical_and(image_true == 3, image_pred == 0)] = 6 # FN FV
    conmap[np.logical_and(image_true == 3, image_pred == 1)] = 5 # FW
    conmap[np.logical_and(image_true == 3, image_pred == 2)] = 5 # FW
    conmap[np.logical_and(image_true == 0, image_pred == 3)] = 7 # FP FV
    conmap[np.logical_and(image_true == 1, image_pred == 3)] = 5 # FW
    conmap[np.logical_and(image_true == 2, image_pred == 3)] = 5 # FW
    conmap[np.logical_and(image_true == 3, image_pred == 3)] = 1 # TP
    if plot_fig:
        fig, ax = plt.subplots()
        cmap = matplotlib.colors.ListedColormap(['white', 'yellow', 'green', 'blue', 'red', 'orange', 'purple', 'lightblue', 'pink'])
        i = ax.imshow(conmap, cmap=cmap, vmin=-1, vmax=7)
        cb = fig.colorbar(i)
        cb.set_ticks(np.arange(-0.5, 7, 8/9))
        cb.set_ticklabels(["NaN", "TN", "TP", "FN", "FP", "FW", "FV FW", "FV FN", "FV FP"])
        return conmap, fig, ax
    return conmap

    
def contingencymap_segments(segments, class_true, class_pred, plot_fig=True):
    """Calculate object-based contingency map/column 
    
    Inputs:
    class_true: str
        Name of column storing truth values.
    class_pred: str
        Name of column storing predicted values.
    plot_fig: bool (default=True)
        If True, a figure is made and fig, ax handles are returned.
    Outputs:
    conmap: pandas Series
        Series of conmap values (-1=NaN, 0=TN, 1=TP, 2=FN, 3=FP, 4=FW, 5=FW, 6=FNFV, 7=FPFV).
    fig, ax: tuple
        Figure and axis handles.
    """
    segments["contin"] = 0
    segments.loc[class_true == -1, "contin"] = -1
    segments.loc[np.logical_and(class_true == 0, class_pred == 1), "contin"] = 3 # FP
    segments.loc[np.logical_and(class_true == 0, class_pred == 2), "contin"] = 3 # FP
    segments.loc[np.logical_and(class_true == 1, class_pred == 0), "contin"] = 2 # FN
    segments.loc[np.logical_and(class_true == 1, class_pred == 1), "contin"] = 1 # TP
    segments.loc[np.logical_and(class_true == 1, class_pred == 2), "contin"] = 4 # FW
    segments.loc[np.logical_and(class_true == 2, class_pred == 0), "contin"] = 2 # FN
    segments.loc[np.logical_and(class_true == 2, class_pred == 1), "contin"] = 4 # FW
    segments.loc[np.logical_and(class_true == 2, class_pred == 2), "contin"] = 1 # TP
    segments.loc[np.logical_and(class_true == 3, class_pred == 0), "contin"] = 6 # FN FV
    segments.loc[np.logical_and(class_true == 3, class_pred == 1), "contin"] = 5 # FW
    segments.loc[np.logical_and(class_true == 3, class_pred == 2), "contin"] = 5 # FW
    segments.loc[np.logical_and(class_true == 0, class_pred == 3), "contin"] = 7 # FP FV
    segments.loc[np.logical_and(class_true == 1, class_pred == 3), "contin"] = 5 # FW
    segments.loc[np.logical_and(class_true == 2, class_pred == 3), "contin"] = 5 # FW
    segments.loc[np.logical_and(class_true == 3, class_pred == 3), "contin"] = 1 # TP
    if plot_fig:
        fig, ax = plt.subplots()
        cmap = matplotlib.colors.ListedColormap(['white', 'yellow', 'green', 'blue', 'red', 'orange', 'purple', 'lightblue', 'pink'])
        i = segments.plot(column="contin", ax=ax, cmap=cmap, vmin=-1, vmax=7)
        cb = fig.colorbar(i)
        cb.set_ticks(np.arange(-0.5, 7, 8/9))
        cb.set_ticklabels(["NaN", "TN", "TP", "FN", "FP", "FW", "FV FW", "FV FN", "FV FP"])
        return segments["contin"], fig, ax
    return segments["contin"]


def calc_accuracy_segments(segments, model_field="model", truth_field="truth", acc_filename=None):
    """ Calculate accuracy of segments
    
    Inputs:
    segments: geopandas GeoDataFrame
        Segments.
    model_field: str (default="model")
        Name of column containing predicted values.
    truth_field: str (default="truth")
        Name of column containing truth values.
    acc_filename: str or None (default=None)
        If not None, accuracy is pickled to specified path.
    Outputs:
    acc: tuple
        F1, Precision, Recall and CSI value.
    """
    model_values = np.array(segments[model_field].copy())
    truth_values = np.array(segments[truth_field].copy())
    model_values[model_values == 3] = 2 # FV considered as wet
    truth_values[truth_values == 3] = 2 # FV considered as wet
    model_values[model_values == 4] = 2 # low lying considered as wet
    acc = calculate_metrics(truth_values, model_values, average="macro", nan_value=-1)
    if acc_filename:
        with open(acc_filename, "wb") as handle:
            pickle.dump(acc, handle)
    return acc

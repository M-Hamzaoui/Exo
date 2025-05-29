#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:17:28 2025

@author: hamzaoui
"""
from sklearn.metrics import confusion_matrix
import numpy as np

def eval_metrics(real_labels,pred_labels):
        tn, fp, fn, tp = confusion_matrix(np.array(real_labels), np.array(pred_labels)).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_acc = (sensitivity + specificity) / 2
        
        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
        HTER = (FAR + FRR) / 2
        return balanced_acc, HTER
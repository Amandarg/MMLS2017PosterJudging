#!/usr/bin/env python
"""sklearn_mle.py: BTL estimation of scores from pairwise comparison data"""
__author__      = "Sumeet Katariya"

import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

def sklearn_mle(X,y):
    """
    Input:
    n: number of images
    df: pandas dataframe containing pairwise comparison data, with headers
    ["left", "right", "winner"]
    Output:
    Learned BTL scores of the images.
    """
    model = LogisticRegression(fit_intercept=False, verbose=True, C=1000000.)
# parameter C controls regularization. High C => less regularization
    model = model.fit(X,y)
    return model.coef_.flatten()

#rankings:
#n: number of total items being compared
def get_X_y(rankings,n):
    num_comparisons = 0
    for r in rankings:
        num_comparisons += len(r)*(len(r)-1)/2
    num_comparisons = num_comparisons

    X = np.zeros((num_comparisons,n))
    y = np.zeros((num_comparisons))

    comparison =0
    for r in rankings:
        for i in range(len(r)-1):
            for j in range(i+1, len(r)):
                lower  = min(r[i], r[j])
                higher = max(r[i], r[j])
                X[comparison, lower]=1
                X[comparison, higher]=-1
                if(lower == r[i]):
                    y[comparison] = 1
                else:
                    y[comparison] = -1
                comparison +=1

    return X, y

"""
Created on April 15, 2018

@author: Alejandro Molina
"""

import numpy as np

from Inference.inference import EPSILON, add_node_likelihood
from Structure.leaves.histogram.Histograms import Histogram

import bisect
import logging

logger = logging.getLogger(__name__)


def histogram_ll(breaks, densities, data):
    probs = np.zeros((data.shape[0], 1))

    for i, x in enumerate(data):
        if x < breaks[0] or x >= breaks[-1]:
            continue

        probs[i] = densities[bisect.bisect(breaks, x) - 1]

    probs[probs < EPSILON] = EPSILON

    return probs


def histogram_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    probs[~marg_ids] = histogram_ll(np.array(node.breaks), np.array(node.densities), nd[~marg_ids])

    return np.log(probs)


def add_histogram_inference_support():
    add_node_likelihood(Histogram, log_lambda_func=histogram_log_likelihood)

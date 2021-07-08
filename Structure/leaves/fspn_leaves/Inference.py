import numpy as np

from Inference.inference import EPSILON, add_node_likelihood
from Structure.leaves.histogram.Histograms import Histogram

import bisect
import logging

logger = logging.getLogger(__name__)


def histogram_ll(breaks, densities, data):
    (n, D) = data.shape
    probs = np.zeros((n, 1))
    for i in range(data.shape[0]):
        x = data[i]
        j = 0
        loc = np.zeros(D)
        while j < D:
            if x[j] < breaks[j][0] or x[j] >= breaks[j][-1]:
                continue
            loc[j] = bisect.bisect(breaks, x[j]) - 1
            j += 1
        p = densities
        for j in loc:
            p = p[j]
        probs[i] = p

    probs[probs < EPSILON] = EPSILON

    return probs


def histogram_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    nd = data[:, list(node.scope)]
    marg_ids = np.isnan(nd).any(axis=1)

    probs[~marg_ids] = histogram_ll(np.array(node.breaks), np.array(node.densities), nd[~marg_ids])

    return np.log(probs)


def add_histogram_inference_support():
    add_node_likelihood(Histogram, log_lambda_func=histogram_log_likelihood)

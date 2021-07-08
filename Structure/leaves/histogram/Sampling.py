"""
Created on May 14, 2019

@author: Alejandro Molina
@author: Zhongjie Yu
"""
from scipy.stats import rv_histogram

from Inference.sampling import add_leaf_sampling
from Structure.StatisticalTypes import MetaType
from Structure.leaves.histogram.Histograms import (
    Histogram
)

import numpy as np
import logging

logger = logging.getLogger(__name__)


def sample_histogram_node(node, n_samples, data, rand_gen):
    assert isinstance(node, Histogram)
    assert n_samples > 0
    # sample the value at each bin according to the densities of each bin
    if node.meta_type == MetaType.DISCRETE or node.meta_type == MetaType.BINARY:
        X = rand_gen.choice(np.array(node.bin_repr_points), p=node.densities, size=n_samples)
    else:
        X = rv_histogram((node.densities, node.breaks)).ppf(rand_gen.random_sample(n_samples))

    return X


def add_histogram_sampling_support():
    add_leaf_sampling(Histogram, sample_histogram_node)


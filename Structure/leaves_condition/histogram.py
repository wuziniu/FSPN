from collections import namedtuple

import numpy as np

from Structure.nodes import Leaf
from Structure.StatisticalTypes import MetaType, Type
from Structure.leaves.histogram.Histograms import create_histogram_leaf, getHistogramVals
import logging

logger = logging.getLogger(__name__)


class Histogram_CPD(Leaf):
    type = Type.CATEGORICAL
    property_type = namedtuple("Histogram_CPD", "breaks densities bin_repr_points")

    def __init__(self, breaks, densities, bin_repr_points, scope=None, condition=None,
                 type_=None, meta_type=MetaType.DISCRETE):
        Leaf.__init__(self, scope=scope, condition=condition)
        self.type = type(self).type if not type_ else type_
        self.meta_type = meta_type
        self.breaks = breaks
        self.densities = densities
        self.bin_repr_points = bin_repr_points
        self.CPD = []

    @property
    def parameters(self):
        return __class__.property_type(
            breaks=self.breaks, densities=self.densities, bin_repr_points=self.bin_repr_points
        )


def create_histogram_condition_leaf(data, ds_context, scope, condition, alpha=1.0):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert len(condition) == 1, "Place reduce conditioned variables"
    assert data.shape[1] == 2, "redundant data"

    data_condition = data[:, 1]
    data_condition = data_condition[~np.isnan(data_condition)]

    idx = condition[0]
    meta_type = ds_context.meta_types[idx]
    domain = ds_context.domains[idx]

    assert not np.isclose(np.max(domain), np.min(domain)), "invalid domain, min and max are the same"

    if data_condition.shape[0] == 0:
        # no data or all were nans
        maxx = np.max(domain)
        minx = np.min(domain)
        breaks = np.array([minx, maxx])
        densities = np.array([1 / (maxx - minx)])
        repr_points = np.array([minx + (maxx - minx) / 2])
        if meta_type == MetaType.DISCRETE or meta_type == MetaType.BINARY:
            repr_points = repr_points.astype(int)

    elif np.var(data) == 0 and meta_type == MetaType.REAL:
        # one data point
        maxx = np.max(domain)
        minx = np.min(domain)
        breaks = np.array([minx, maxx])
        densities = np.array([1 / (maxx - minx)])
        repr_points = np.array([minx + (maxx - minx) / 2])
        if meta_type == MetaType.DISCRETE or meta_type == MetaType.BINARY:
            repr_points = repr_points.astype(int)

    else:
        breaks, densities, repr_points = getHistogramVals(data_condition, meta_type, domain)

    # laplace smoothing
    if alpha:
        n_samples = data_condition.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert len(densities) == len(breaks) - 1

    leaf = Histogram_CPD(breaks.tolist(), densities.tolist(), repr_points.tolist(), scope=idx, meta_type=meta_type)
    data_slices = slice_data_on_condition(data, breaks.tolist(), densities.tolist())
    for data_slice in data_slices:
        if len[data_slices] == 0:
            leaf.CPD.append(None)
        else:
            leaf.CPD.append(create_histogram_leaf(data_slice, ds_context, scope))

    return leaf


def slice_data_on_condition(data, breaks, densities):
    """
    split the data of scope based on the value of condition
    """
    result = []
    for i in range(len(densities)):
        if densities[i] == 0:
            result.append([])
        else:
            slice_idx = np.intersect1d(np.where(breaks[i] <= data[1]), np.where(data[1] < breaks[i+1]))
            slice = data[0, slice_idx]
            result.append(slice)
    return result



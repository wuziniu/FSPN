from collections import namedtuple

import numpy as np

from Structure.nodes import Leaf
from Structure.StatisticalTypes import MetaType, Type
import logging

logger = logging.getLogger(__name__)


class Histogram_CPD(Leaf):

    type = Type.CATEGORICAL
    property_type = namedtuple("Histogram", "breaks densities bin_repr_points")

    def __init__(self, breaks, densities, bin_repr_points, scope=None, condition=None,
                 type_=None, meta_type=MetaType.DISCRETE):
        Leaf.__init__(self, scope=scope, condition=condition)
        self.type = type(self).type if not type_ else type_
        self.meta_type = meta_type
        self.breaks = breaks
        self.densities = densities
        self.bin_repr_points = bin_repr_points

    @property
    def parameters(self):
        return __class__.property_type(
            breaks=self.breaks, densities=self.densities, bin_repr_points=self.bin_repr_points
        )




def create_histogram_condition_leaf(data, ds_context, scope, condition, alpha=1.0):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == len(scope)+len(condition), "redundant data"

    data = data[~np.isnan(data)]

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]
    domain = ds_context.domains[idx]

    assert not np.isclose(np.max(domain), np.min(domain)), "invalid domain, min and max are the same"

    if data.shape[0] == 0:
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
        breaks, densities, repr_points = getHistogramVals(data, meta_type, domain)

    # laplace smoothing
    if alpha:
        n_samples = data.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert len(densities) == len(breaks) - 1

    return Histogram(breaks.tolist(), densities.tolist(), repr_points.tolist(), scope=idx, meta_type=meta_type)


def getHistogramVals(data, meta_type, domain):
    # check this: https://github.com/theodoregoetz/histogram

    if meta_type == MetaType.DISCRETE or meta_type == MetaType.BINARY:
        # for discrete, we just have to count
        breaks = np.array([d for d in domain] + [domain[-1] + 1])
        densities, breaks = np.histogram(data, bins=breaks, density=True)
        repr_points = np.asarray(domain)
        return breaks, densities, repr_points

    densities, breaks = np.histogram(data, bins="auto", density=True)
    mids = ((breaks + np.roll(breaks, -1)) / 2.0)[:-1]
    return breaks, densities, mids



from collections import namedtuple
import numpy as np
from Inference.inference import EPSILON

from Structure.nodes import Leaf
from Structure.StatisticalTypes import MetaType, Type
from Structure.leaves.get_breaks import get_breaks
import logging

logger = logging.getLogger(__name__)

class Histogram(Leaf):

    type = Type.CATEGORICAL
    property_type = namedtuple("Histogram", "breaks pdf cdf")

    def __init__(self, breaks, pdf, cdf, nan_perc=1.0, scope=None, type_=None, meta_type=MetaType.DISCRETE):
        Leaf.__init__(self, scope=scope)
        self.type = type(self).type if not type_ else type_
        self.meta_type = meta_type
        self.breaks = breaks
        self.pdf = pdf
        assert np.isclose(np.sum(pdf), 1), "incorrect pdf"
        self.cdf = cdf
        assert np.isclose(cdf[-1], 1), "incorrect cdf"
        self.nan_perc = nan_perc

    @property
    def parameters(self):
        return __class__.property_type(
            breaks=self.breaks, pdf=self.pdf, cdf=self.cdf
        )

    def query(self, query, attr):
        if type(query) == tuple:
            if set(attr) != set(self.scope):
                query_idx = [attr.index(i) for i in self.scope]
                query = (query[0][:, query_idx], query[1][:, query_idx])
            return self.infer_range_query(query)
        else:
            if set(attr) != set(self.scope):
                query_idx = [attr.index(i) for i in self.scope]
                query = query[:, query_idx]
            return self.infer_point_query(query)

    def infer_point_query(self, query, epsilon=False):
        n = query.shape[0]
        probs = np.zeros(n)
        assert query.shape[1] == 1, "use multivariate histogram"

        breaks = self.breaks  # adding an illegal value bin
        idx = np.searchsorted(breaks, query[:, 0])
        legal = np.where((idx > 0) & (idx < len(breaks)))
        probs[legal] = self.pdf[idx[legal] - 1]
        probs *= self.nan_perc
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        return probs

    def infer_range_query(self, query, epsilon=False):
        left_bound = query[0]
        right_bound = query[1]
        n = left_bound.shape[0]
        probs = np.zeros(n)
        assert left_bound.shape[1] == 1, "use multivariate histogram"

        breaks = self.breaks
        l_idx = np.searchsorted(breaks, left_bound[:, 0])
        r_idx = np.searchsorted(breaks, right_bound[:, 0])
        legal = np.where((l_idx < len(breaks)) & (r_idx > 0))
        l_idx[l_idx == 0] = 1
        r_idx[r_idx == len(breaks)] = len(breaks) - 1
        probs[legal] = self.cdf[r_idx[legal]] - self.cdf[l_idx[legal]-1]
        probs *= self.nan_perc
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        else:
            probs[probs < 0] = 0
        return probs


def create_histogram_leaf(data, ds_context, scope, condition, alpha=1.0, hist_source="numpy", discretize=False):
    assert len(scope) + len(condition) == data.shape[1], "redundant data"
    assert len(scope) == 1, "use Multi_histogram for more than two values"
    idx = sorted(scope + condition)
    keep = []
    for i in range(len(idx)):
        if idx[i] in scope:
            keep.append(i)
    data = data[:, keep]

    n = len(data)
    data = data[~np.isnan(data)]
    nan_perc = len(data)/n

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]
    domain = ds_context.domains[idx]

    assert not np.isclose(np.max(domain), np.min(domain)), "invalid domain, min and max are the same"

    if data.shape[0] == 0:
        # no data or all were nans
        maxx = np.max(domain)
        minx = np.min(domain)
        breaks = np.array([minx, maxx])
        densities = np.array([0])
        cdf = np.array([0, 0])

    elif np.var(data) == 0 and meta_type == MetaType.REAL:
        # one data point
        breaks = np.array([data[0]-EPSILON/100, data[0]])
        densities = np.array([1.])
        cdf = np.array([0., 1.])

    else:
        breaks, densities, cdf = getHistogramVals(data, meta_type, domain, source=hist_source,
                                                  discretize=discretize)
    # laplace smoothing
    if alpha:
        n_samples = data.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)
        cdf = np.zeros(len(densities)+1)
        cdf[1:] = np.cumsum(densities)

    assert len(densities) == len(breaks) - 1

    return Histogram(breaks.tolist(), densities, cdf, nan_perc=nan_perc, scope=scope, meta_type=meta_type)


def getHistogramVals(data, meta_type, domain, num_bins=60, source="numpy", discretize=False):
    # check this: https://github.com/theodoregoetz/histogram

    if meta_type == MetaType.DISCRETE or meta_type == MetaType.BINARY or discretize == False:
        # for discrete, we just have to count
        breaks = get_breaks(data, domain)
        pdf, breaks = np.histogram(data, bins=breaks)
        pdf = pdf/len(data)
        cdf = np.zeros(len(pdf)+1)
        for i in range(len(pdf)):
            if i == 0:
                cdf[i+1] = pdf[i]
            else:
                cdf[i+1] = pdf[i]+cdf[i]
        return breaks, pdf, cdf

    if source == "kde":
        import statsmodels.api as sm

        kde = sm.nonparametric.KDEMultivariate(data, var_type="c", bw="cv_ls")
        bins = int((domain[1] - domain[0]) / kde.bw)
        bins = min(num_bins, bins)
        cdf_x = np.linspace(domain[0], domain[1], 2 * bins)
        cdf_y = kde.cdf(cdf_x)
        breaks = np.interp(np.linspace(0, 1, bins), cdf_y, cdf_x)  # inverse cdf
        mids = ((breaks + np.roll(breaks, -1)) / 2.0)[:-1]
        densities = kde.pdf(mids)
        densities / np.sum(densities)
        pdf = densities
        if len(densities.shape) == 0:
            pdf = np.array([densities])
        cdf = np.cumsum(densities)
        return breaks, pdf, cdf

    if source == "numpy":
        #discretize continous domain
        pdf, breaks = np.histogram(data, bins="auto")       
        pdf = pdf / len(data)
        assert np.isclose(np.sum(pdf), 1), "incorrect pdf"
        cdf = np.cumsum(pdf)
        return breaks, pdf, cdf


    assert False, "unkown histogram method " + source

from collections import namedtuple
from Inference.inference import EPSILON
import numpy as np

from Structure.nodes import Leaf
from Structure.StatisticalTypes import MetaType, Type
from Structure.leaves.get_breaks import get_breaks
import logging
from Structure.leaves.fspn_leaves.dimension_reduction import PCA_reduction
from Structure.leaves.fspn_leaves.utils import discretize_series

logger = logging.getLogger(__name__)
rpy_initialized = False


class Multi_histogram(Leaf):

    type = Type.CATEGORICAL
    property_type = namedtuple("Multi_Histogram", "breaks pdf cdf")

    def __init__(self, breaks, pdf, cdf, nan_perc=1.0, scope=None, type_=None, meta_type=MetaType.DISCRETE,
                 red_machine=None):
        Leaf.__init__(self, scope=scope)
        self.type = type(self).type if not type_ else type_
        self.meta_type = meta_type
        self.breaks = breaks
        self.pdf = pdf
        self.cdf = cdf
        self.nan_perc = nan_perc
        self.red_machine=None

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
            if self.red_machine:
                return self.infer_range_query_with_reduction(query)
            else:
                return self.infer_range_query(query)
        else:
            if set(attr) != set(self.scope):
                query_idx = [attr.index(i) for i in self.scope]
                query = query[:, query_idx]
            return self.infer_point_query(query)


    def infer_point_query(self, query, epsilon=False):
        n = query.shape[0]
        probs = np.zeros(n)
        assert query.shape[1] != 1, "use univariate histogram"

        s = None
        idx = []
        for i, breaks in enumerate(self.breaks):
            idx.append(np.searchsorted(breaks, query[:, i]))
            if i == 0:
                s = (idx[i] > 0) & (idx[i] < len(breaks))
            else:
                s = s & (idx[i] > 0) & (idx[i] < len(breaks))
        legal = np.where(s)
        query_idx = []
        for i in range(len(self.breaks)):
            idx_i = idx[i][legal] - 1
            query_idx.append(idx_i)
        probs[legal] = self.pdf[tuple(query_idx)]
        probs *= self.nan_perc
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        return probs

    def infer_range_query_with_reduction(self, query, epsilon=False):
        assert self.red_machine is not None, "Did not performed dimension reduction"
        batch_size = query[0].shape[0]
        query = np.concatenate(query, axis=0)
        query_red = self.red_machine.transform(query)
        query_red = (query_red[0:batch_size, :], query_red[batch_size:, :])
        if self.red_machine.n_components == 1:
            return self.infer_range_query_single(query_red, epsilon)
        else:
            return self.infer_range_query(query_red, epsilon)
            

    def infer_range_query_single(self, query, epsilon=False):
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
        probs[legal] = self.cdf[r_idx[legal]] - self.cdf[l_idx[legal] - 1]
        probs *= self.nan_perc
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        else:
            probs[probs < 0] = 0
        return probs

    def infer_range_query(self, query, epsilon=False):
        left_bound = query[0]
        right_bound = query[1]
        n = left_bound.shape[0]
        probs = np.zeros(n)
        assert left_bound.shape[1] != 1, "use univariate histogram"

        s = None
        idx_l = []
        idx_r = []
        for i, breaks in enumerate(self.breaks):
            idx_l.append(np.searchsorted(breaks, left_bound[:, i]))
            idx_r.append(np.searchsorted(breaks, right_bound[:, i]))
            if i == 0:
                s = (idx_l[i] < len(breaks)) & (idx_r[i] > 0)
            else:
                s = s & (idx_l[i] < len(breaks)) & (idx_r[i] > 0)
            s = s & (idx_l[i] <= idx_r[i])

        legal = np.where(s)[0]
        query_idx_l = np.ones((len(self.breaks), len(legal)))
        query_idx_r = np.ones((len(self.breaks), len(legal)))
        for i in range(len(self.breaks)):
            idx_li = idx_l[i][legal]
            idx_li[idx_li == 0] = 1
            idx_li = idx_li-1
            idx_ri = idx_r[i][legal]
            query_idx_l[i, :] = idx_li
            query_idx_r[i, :] = idx_ri
        query_idx_l = query_idx_l.T.astype(int)
        query_idx_r = query_idx_r.T.astype(int)
        d = len(self.breaks)
        legal_idx = list(legal)
        for i, idx in enumerate(legal_idx):
            l = tuple(query_idx_l[i])
            r = tuple(query_idx_r[i])
            # this is sooooooooo stupid!!!!!! I wish someone can give me an efficient solution
            if d == 2:
                probs[idx] = np.sum(self.pdf[l[0]:r[0], l[1]:r[1]])
            elif d == 3:
                probs[idx] = np.sum(self.pdf[l[0]:r[0], l[1]:r[1], l[2]:r[2]])
            elif d == 4:
                probs[idx] = np.sum(self.pdf[l[0]:r[0], l[1]:r[1], l[2]:r[2], l[3]:r[3]])
            elif d == 5:
                probs[idx] = np.sum(self.pdf[l[0]:r[0], l[1]:r[1], l[2]:r[2], l[3]:r[3], l[4]:r[4]])
            elif d == 6:
                probs[idx] = np.sum(self.pdf[l[0]:r[0], l[1]:r[1], l[2]:r[2], l[3]:r[3], l[4]:r[4], l[5]:r[5]])
            elif d == 7:
                probs[idx] = np.sum(self.pdf[l[0]:r[0], l[1]:r[1], l[2]:r[2], l[3]:r[3], l[4]:r[4], l[5]:r[5],
                                    l[6]:r[6]])
            else:
                assert False, "implement more if statement????? Lolllllll"

        probs *= self.nan_perc
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        else:
            probs[probs < 0] = 0
        return probs


    def infer_range_query_fancy(self, query, epsilon=False):
        left_bound = query[0]
        right_bound = query[1]
        n = left_bound.shape[0]
        probs = np.zeros(n)
        assert left_bound.shape[1] != 1, "use univariate histogram"

        s = None
        idx_l = []
        idx_r = []
        for i, brk in enumerate(self.breaks):
            breaks = [brk[0] - 1] + brk  # adding an illegal value bin
            idx_l.append(np.searchsorted(breaks, left_bound[:, i]))
            idx_r.append(np.searchsorted(breaks, right_bound[:, i]))
            if i == 0:
                s = (idx_l[i] < len(breaks)) & (idx_r[i] > 0)
            else:
                s = s & (idx_l[i] < len(breaks)) & (idx_r[i] > 0)
            s = s & (idx_l[i] < idx_r[i])
        legal = np.where(s)
        query_idx_l = []
        query_idx_r = []
        for i in range(len(self.breaks)):
            idx_li = idx_l[i][legal] - 1
            idx_ri = idx_r[i][legal] - 1
            query_idx_l.append(idx_li)
            query_idx_r.append(idx_ri)
        probs[legal] = self.cdf[tuple(query_idx_r)] - self.cdf[tuple(query_idx_l)]
        probs *= self.nan_perc
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        else:
            probs[probs < 0] = 0
        return probs.reshape

    def expectation(self, query, fanouts, attr):
        return self.query(query, attr)


def multidim_cumsum(a):
    """
    This function calculates the cdf of multi-dimensional pdf
    """
    out = a.cumsum(-1)
    for i in range(2, a.ndim+1):
        np.cumsum(out, axis=-i, out=out)
    return out

def create_multi_histogram_leaf(data, ds_context, scope, condition, alpha=False, discretize=False,
                                dim_red=False, n_mcv=0, n_bins=80):
    assert len(scope)+len(condition) == data.shape[1], "redundant data"
    idx = sorted(scope + condition)
    keep = []
    for i in range(len(idx)):
        if idx[i] in scope:
            keep.append(i)
    data = data[:, keep]
    n = len(data)
    data = data[~np.isnan(data).any(axis=1)]
    nan_perc = len(data) / n

    if data.shape[0] == 0:
        # no data or all were nans, just return 0 whenever queried
        breaks = []
        pdf = np.array([0])
        pdf.reshape(tuple([1]*len(scope)))
        cdf = pdf
        return Multi_histogram(breaks, pdf, cdf, nan_perc, scope=scope)

    if dim_red:
        pca, data = PCA_reduction(data)
    else:
        pca = None

    if pca:
        logger.info(f"reduced the dimension from {len(scope)} to {pca.n_components}")
        breaks = []
        break_size = []
        for i in range(data.shape[1]):
            _, cont_break = discretize_series(data[:, i], n_mcv, n_bins)
            breaks.append(np.array(cont_break))
            break_size.append(n_mcv+n_bins)
        logger.info(f"Multihistogram of size {break_size}")
        if len(breaks) == 1:
            pdf, breaks = np.histogram(data, bins=breaks[0])
            pdf = pdf / len(data)
            assert np.isclose(np.sum(pdf), 1), "incorrect pdf"
            cdf = np.cumsum(pdf)
        else:
            pdf, breaks = np.histogramdd(data, bins=breaks)
            pdf = pdf / np.sum(pdf)
            cdf = multidim_cumsum(pdf)
    else:
        breaks = []
        break_size = []
        for i, s in enumerate(scope):
            meta_type = ds_context.meta_types[s]
            domain = ds_context.domains[s]
            assert not np.isclose(np.max(domain), np.min(domain)), "invalid domain, min and max are the same"
            if meta_type == MetaType.BINARY or meta_type == MetaType.DISCRETE or discretize == False:
                attr_breaks = get_breaks(data[:, i], domain)
                break_size.append(len(attr_breaks))
                breaks.append(attr_breaks)
            else:
                _, cont_break = np.histogram(data[:, i], bins="auto")
                break_size.append(len(cont_break)-1)
                breaks.append(np.array(cont_break))
        logger.info(f"Multihistogram of size {break_size}")
        pdf, breaks = np.histogramdd(data, bins=breaks)
        pdf = pdf/np.sum(pdf)
        cdf = multidim_cumsum(pdf)

    if alpha:
        logger.warning("Can we smooth a multidimensional histogram?")

    return Multi_histogram(breaks, pdf, cdf, nan_perc, scope=scope, red_machine=pca)

from collections import namedtuple
import numpy as np

from Structure.nodes import Leaf
from Structure.StatisticalTypes import MetaType, Type
import logging

logger = logging.getLogger(__name__)

EPSILON = np.finfo(float).eps
class Binary(Leaf):

    type = Type.CATEGORICAL
    property_type = namedtuple("Binary", "pdf")

    def __init__(self, pdf, nan_perc=1.0, scope=None, type_=None, meta_type=MetaType.DISCRETE):
        Leaf.__init__(self, scope=scope)
        self.type = type(self).type if not type_ else type_
        self.meta_type = meta_type
        self.breaks = []
        self.cdf = []
        self.pdf = pdf
        assert np.isclose(np.sum(pdf), 1), "incorrect pdf"
        self.nan_perc = nan_perc

    @property
    def parameters(self):
        return __class__.property_type(
            pdf=self.pdf
        )

    def likelihood(self, data, attr, log=False):
        if set(attr) != set(self.scope):
            data_idx = [attr.index(i) for i in self.scope]
            data = data[:, data_idx]

        ll = self.pdf[data] * self.nan_perc
        ll = ll.reshape(len(data))
        ll[ll < EPSILON] = EPSILON
        if log:
            return np.log(ll)
        return ll


def create_binary_leaf(data, ds_context, scope, condition):
    assert len(scope) + len(condition) == data.shape[1], "redundant data"
    assert len(scope) == 1, "use Multi_binary for more than two values"
    idx = sorted(scope + condition)
    keep = []
    for i in range(len(idx)):
        if idx[i] in scope:
            keep.append(i)
    data = data[:, keep]

    n = len(data)
    data = data[~np.isnan(data)]
    nan_perc = len(data)/n
    pdf = [np.sum(data == 0)/len(data), np.sum(data == 1)/len(data)]
    return Binary(np.asarray(pdf), nan_perc, scope)

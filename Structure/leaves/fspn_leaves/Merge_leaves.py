from collections import namedtuple
from Inference.inference import EPSILON
import numpy as np

from Structure.nodes import Leaf
from Structure.StatisticalTypes import MetaType, Type
import logging

logger = logging.getLogger(__name__)

class Merge_leaves(Leaf):

    type = Type.CATEGORICAL
    property_type = namedtuple("Merge_leaves", "scope condition range")

    def __init__(self, leaves, scope=None, condition=None, ranges=None, type_=None, meta_type=MetaType.DISCRETE):
        Leaf.__init__(self, scope=scope)
        self.type = type(self).type if not type_ else type_
        self.meta_type = meta_type
        self.scope = scope
        self.condition = condition
        self.range = ranges
        self.leaves = leaves

    @property
    def parameters(self):
        return __class__.property_type(
            breaks=self.leaves
        )

    def query(self, query, attr):
        if type(query) == tuple:
            return self.infer_range_query(query)
        else:
            return self.infer_point_query(query)

    def infer_point_query(self, query, epsilon=False):
        children_res = []
        for leaf in self.leaves:
            idx = [self.scope.index(i) for i in leaf.scope]
            leaf_query = query[:, idx]
            prob = leaf.infer_point_query(leaf_query, epsilon)
            children_res.append(prob)
        llchildren = np.stack(children_res, axis=1)
        probs = np.prod(llchildren, axis=1)
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        else:
            probs[probs < 0] = 0
        return probs


    def infer_range_query(self, query, epsilon=False):
        children_res = []
        for leaf in self.leaves:
            idx = [self.scope.index(i) for i in leaf.scope]
            leaf_query = (query[0][:, idx], query[1][:, idx])
            prob = leaf.infer_range_query(leaf_query, epsilon)
            children_res.append(prob)
        llchildren = np.stack(children_res, axis=1)
        probs = np.prod(llchildren, axis=1)
        if epsilon:
            probs[probs < EPSILON] = EPSILON
        else:
            probs[probs < 0] = 0
        return probs

    def likelihood(self, data, attr, log=False):
        children_ll = []
        for leaf in self.leaves:
            ll = leaf.likelihood(data, attr, log)
            children_ll.append(ll)
        llchildren = np.stack(children_ll, axis=1)
        if log:
            return np.sum(llchildren, axis=1).reshape(-1)
        return np.prod(llchildren, axis=1).reshape(-1)


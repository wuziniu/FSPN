from copy import deepcopy

from Learning.transformStructure import Prune
from Learning.validity import is_valid
from Structure.nodes import Sum, Leaf, Factorize, assign_ids
import logging

logger = logging.getLogger(__name__)


def marginalize(node, keep):
    # keep must be a set of features that you want to keep

    keep = set(keep)

    def marg_recursive(node):
        new_node_scope = keep.intersection(set(node.scope))

        if len(new_node_scope) == 0:
            # we are summing out this node
            return None

        if isinstance(node, Leaf):
            return deepcopy(node)

        newNode = node.__class__()

        if isinstance(node, Sum):
            newNode.weights.extend(node.weights)

        for c in node.children:
            new_c = marg_recursive(c)
            if new_c is None:
                continue
            newNode.children.append(new_c)

        newNode.scope.extend(new_node_scope)
        return newNode

    newNode = marg_recursive(node)
    assign_ids(newNode)
    newNode = Prune(newNode)
    valid, err = is_valid(newNode)
    assert valid, err

    return newNode
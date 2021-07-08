import logging
import numpy as np
from Structure.nodes import Product, Sum, Factorize, Leaf, get_topological_order

logger = logging.getLogger(__name__)

EPSILON = np.finfo(float).eps


def prod_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype
    return np.prod(llchildren, axis=1).reshape(-1, 1)


def sum_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    return np.dot(llchildren, b).reshape(-1, 1)

def factorize_likelihood(node, r_children, l_children, dtype=np.float64, **kwargs):
    assert len(r_children) == len(l_children), "probability shape mismatch"
    r_children = np.concatenate(r_children, axis=1)
    l_children = np.concatenate(l_children, axis=1).transpose()
    assert r_children.dtype == dtype
    assert l_children.dtype == dtype

    return np.dot(r_children, l_children).reshape(-1, 1)

_node_likelihood = {Sum: sum_likelihood, Product: prod_likelihood, Factorize: factorize_likelihood}


def likelihood(node, data, dtype=np.float64, node_likelihood=_node_likelihood, lls_matrix=None, debug=False, **kwargs):
    all_results = {}

    if debug:
        assert len(data.shape) == 2, "data must be 2D, found: {}".format(data.shape)
        original_node_likelihood = node_likelihood

        def exec_funct(node, *args, **kwargs):
            assert node is not None, "node is nan "
            funct = original_node_likelihood[type(node)]
            ll = funct(node, *args, **kwargs)
            assert ll.shape == (data.shape[0], 1), "node %s result has to match dimensions (N,1)" % node.id
            assert not np.any(np.isnan(ll)), "ll is nan %s " % node.id
            return ll

        node_likelihood = {k: exec_funct for k in node_likelihood.keys()}

    result = eval_spn_bottom_up(node, node_likelihood, all_results=all_results, debug=debug, dtype=dtype, data=data,
                                **kwargs)

    if lls_matrix is not None:
        for n, ll in all_results.items():
            lls_matrix[:, n.id] = ll[:, 0]

    return result


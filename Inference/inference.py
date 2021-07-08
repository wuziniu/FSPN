import logging
import numpy as np
from scipy.special import logsumexp

from Structure.nodes import Product, Sum, Factorize, eval_spn_bottom_up

logger = logging.getLogger(__name__)

EPSILON = 0.1


def leaf_marginalized_likelihood(node, data=None, dtype=np.float64, log_space=False, **kwargs):
    #assert len(node.scope) == 1, node.scope
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    if log_space:
        probs[:] = 0
    assert data.shape[1] >= 1
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]
    assert len(observations.shape) == 1, observations.shape
    return probs, marg_ids, observations


def prod_log_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.stack(children, axis=1)
    assert llchildren.dtype == dtype
    pll = np.sum(llchildren, axis=1).reshape(-1, 1)
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min

    return pll.reshape(-1)


def prod_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.stack(children, axis=1)
    assert llchildren.dtype == dtype
    return np.prod(llchildren, axis=1)


def sum_log_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.stack(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    sll = logsumexp(llchildren, b=b, axis=1)

    return sll.reshape(-1)


def sum_likelihood(node, children, dtype=np.float64, **kwargs):
    llchildren = np.stack(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    return np.dot(llchildren, b)

def factorize_likelihood(node, r_children, l_children, dtype=np.float64, **kwargs):
    assert len(r_children) == len(l_children), "probability shape mismatch"
    r_children = np.stack(r_children, axis=1)
    l_children = np.stack(l_children, axis=1)
    assert r_children.dtype == dtype
    assert l_children.dtype == dtype

    return (r_children * l_children).reshape(-1, 1)

def factorize_log_likelihood(node, r_children, l_children, dtype=np.float64, **kwargs):
    assert len(r_children) == len(l_children), "probability shape mismatch"
    r_children = np.stack(r_children, axis=1)
    l_children = np.stack(l_children, axis=1)
    assert r_children.dtype == dtype
    assert l_children.dtype == dtype
    pll = (r_children + l_children).reshape(-1, 1)
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min
    return pll

_node_log_likelihood = {Sum: sum_log_likelihood, Product: prod_log_likelihood, Factorize: factorize_log_likelihood}
_node_likelihood = {Sum: sum_likelihood, Product: prod_likelihood, Factorize: factorize_likelihood}


def _get_exp_likelihood(f_log):
    def f_exp(node, *args, **kwargs):
        return np.exp(f_log(node, *args, **kwargs))

    return f_exp


def _get_log_likelihood(f_exp):
    def f_log(node, *args, **kwargs):
        with np.errstate(divide="ignore"):
            nll = np.log(f_exp(node, *args, **kwargs))
            nll[np.isinf(nll)] = np.finfo(nll.dtype).min
            assert not np.any(np.isnan(nll))
            return nll

    return f_log


def add_node_likelihood(node_type, lambda_func=None, log_lambda_func=None):
    assert not (lambda_func is None and log_lambda_func is None)

    if lambda_func is None:
        lambda_func = _get_exp_likelihood(log_lambda_func)
    _node_likelihood[node_type] = lambda_func

    if log_lambda_func is None:
        log_lambda_func = _get_log_likelihood(lambda_func)
    _node_log_likelihood[node_type] = log_lambda_func


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


def log_likelihood(
        node, data, dtype=np.float64, node_log_likelihood=_node_log_likelihood, lls_matrix=None, debug=False, **kwargs):
    return likelihood(node, data, dtype=dtype, node_likelihood=node_log_likelihood, lls_matrix=lls_matrix, debug=debug,
                      **kwargs)


def conditional_log_likelihood(node_joint, node_marginal, data, log_space=True, dtype=np.float64):
    result = log_likelihood(node_joint, data, dtype) - log_likelihood(node_marginal, data, dtype)
    if log_space:
        return result

    return np.exp(result)



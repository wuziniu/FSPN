import logging
import time
import copy
from Learning.utils import convert_to_scope_domain
from sklearn.cluster import KMeans
from Structure.leaves.fspn_leaves.Multi_Histograms import Multi_histogram, multidim_cumsum
from Structure.leaves.fspn_leaves.Histograms import Histogram
from Structure.leaves.fspn_leaves.Merge_leaves import Merge_leaves

from Inference.inference import EPSILON

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time
from Learning.splitting.RDC import rdc_test
from Structure.nodes import Product, Sum, Factorize, Leaf

parallel = True

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_RDC(data, ds_context, scope, condition, sample_size):
    """
    Calculate the RDC adjacency matrix using the data
    """
    tic = time.time()
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    meta_types = ds_context.get_meta_types_by_scope(scope_range)
    domains = ds_context.get_domains_by_scope(scope_range)

    # calculate the rdc scores, the parameter k to this function are taken from SPFlow original code
    if len(data) <= sample_size:
        rdc_adjacency_matrix = rdc_test(
            data, meta_types, domains, k=10
        )
    else:
        local_data_sample = data[np.random.randint(data.shape[0], size=sample_size)]
        rdc_adjacency_matrix = rdc_test(
            local_data_sample, meta_types, domains, k=10
        )
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    logging.debug(f"calculating pairwise RDC on sample {sample_size} takes {time.time() - tic} secs")
    return rdc_adjacency_matrix, scope_loc, condition_loc


def top_down_update(fspn, dataset, ds_context):
    """
        Updates the FSPN when a new dataset arrives. The function recursively traverses the
        tree and inserts the different values of a dataset at the according places.
        At every sum node, the child node is selected, based on the minimal euclidian distance to the
        cluster_center of on of the child-nodes.
    """
    if fspn.range:
        assert dataset.shape[1] == len(fspn.scope) + len(fspn.range), \
            f"mismatched data shape {dataset.shape[1]} and {len(fspn.scope) + len(fspn.condition)}"
    else:
        assert dataset.shape[1] == len(fspn.scope), \
            f"mismatched data shape {dataset.shape[1]} and {len(fspn.scope)}"

    if isinstance(fspn, Leaf):
        # TODO: indepence test along with original_dataset for multi-leaf nodes
        update_leaf(fspn, dataset, ds_context)
        return None

    elif isinstance(fspn, Factorize):
        index_list = sorted(fspn.scope + fspn.condition)
        left_cols = [fspn.scope.index(i) for i in fspn.children[0].scope]
        top_down_update(fspn.children[0], dataset[:, left_cols], ds_context)
        top_down_update(fspn.children[1], dataset, ds_context)

    elif isinstance(fspn, Sum) and fspn.range is not None:
        # a split node
        assert fspn.cluster_centers == [], fspn
        total_len = 0
        for child in fspn.children:
            assert child.range is not None, child
            new_data = split_data_by_range(dataset, child.range, child.scope)
            total_len += len(new_data)
            top_down_update(child, new_data, ds_context)
        # assert np.sum(total_len) == len(dataset), f"{np.sum(total_len)} and {len(dataset)}"

    elif isinstance(fspn, Sum):
        # a sum node
        assert len(fspn.cluster_centers) != 0, fspn
        total_len = 0
        new_data = split_data_by_cluster_center(dataset, fspn.cluster_centers)
        origin_cardinality = fspn.cardinality
        fspn.cardinality += len(dataset)
        for i, child in enumerate(fspn.children):
            dl = len(new_data[i])
            total_len += dl
            child_cardinality = origin_cardinality * fspn.weights[i]
            child_cardinality += dl
            fspn.weights[i] = child_cardinality / fspn.cardinality
            top_down_update(child, new_data[i], ds_context)
        assert total_len == len(dataset), "ambiguous data point exists"

    elif isinstance(fspn, Product):
        # TODO: indepence test along with original_dataset
        for child in fspn.children:
            index = [fspn.scope.index(s) for s in child.scope]
            top_down_update(child, dataset[:, index], ds_context)


def update_leaf(fspn, dataset, ds_context):
    """
    update the parameter of leaf distribution, currently only support histogram.
    """
    if isinstance(fspn, Histogram):
        update_leaf_Histogram(fspn, dataset, ds_context)
    elif isinstance(fspn, Multi_histogram):
        update_leaf_Multi_Histogram(fspn, dataset, ds_context)
    elif isinstance(fspn, Merge_leaves):
        update_leaf_Merge(fspn, dataset, ds_context)
    else:
        # TODO: implement the update of other leaf nodes
        assert False, "update of other node type is not yet implemented!!!!"


def update_leaf_Histogram(fspn, dataset, ds_context):
    """
    Insert the new data into the original histogram and update the parameter.
    """
    if fspn.range:
        if len(fspn.scope + fspn.condition) == dataset.shape[1]:
            idx = sorted(fspn.scope + fspn.condition)
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        elif len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]:
            idx = sorted(fspn.scope + list(fspn.range.keys()))
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        else:
            assert False

    new_card = len(dataset)
    if new_card == 0:
        return
    dataset = dataset[~np.isnan(dataset)]
    new_card_actual = len(dataset)  # the cardinality without nan.
    new_nan_perc = new_card_actual / new_card

    old_card = fspn.cardinality
    fspn.cardinality = old_card + new_card
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_weight = old_card / (new_card + old_card)
    new_weight = new_card / (new_card + old_card)
    fspn.nan_perc = old_weight * fspn.nan_perc + new_weight * new_nan_perc  # update nan_perc

    if new_card_actual == 0:
        return
    old_weight = old_card_actual / (new_card_actual + old_card_actual)
    new_weight = new_card_actual / (new_card_actual + old_card_actual)

    new_breaks = list(fspn.breaks)
    left_added = False
    right_added = False
    # new value out of bound of original breaks, adding new break
    if np.min(dataset) < new_breaks[0]:
        new_breaks = [np.min(dataset) - EPSILON] + new_breaks
        left_added = True
    if np.max(dataset) > new_breaks[-1]:
        new_breaks = new_breaks + [np.max(dataset) + EPSILON]
        right_added = True

    new_pdf, new_breaks = np.histogram(dataset, bins=new_breaks)
    new_pdf = new_pdf / np.sum(new_pdf)
    old_pdf = fspn.pdf.tolist()
    if left_added:
        old_pdf = [0.0] + old_pdf
    if right_added:
        old_pdf = old_pdf + [0.0]
    old_pdf = np.asarray(old_pdf)

    assert len(new_pdf) == len(old_pdf) == len(new_breaks) - 1, "lengths mismatch"
    new_pdf = old_pdf * old_weight + new_pdf * new_weight
    new_cdf = np.zeros(len(new_pdf) + 1)
    for i in range(len(new_pdf)):
        if i == 0:
            new_cdf[i + 1] = new_pdf[i]
        else:
            new_cdf[i + 1] = new_pdf[i] + new_cdf[i]
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"
    assert np.isclose(new_cdf[-1], 1), f"incorrect cdf, with max {new_cdf[-1]}"

    fspn.breaks = new_breaks
    fspn.pdf = new_pdf
    fspn.cdf = new_cdf

    """
    # a different implementation goes as follow
    idx = np.searchsorted(fspn.breaks, dataset)  #search where to insert into the histogram
    left_most = np.where(idx > 0)[0]  #new value out of left bound, adding new breaks
    right_most = np.where(idx < len(fspn.breaks))[0]
    legal = np.where((idx > 0) & (idx < len(fspn.breaks)))[0]
    new_pdf = list(fspn.pdf)
    for i in np.unique(idx[legal]):
        new_prob = len(np.where(idx == i)[0])/new_card_actual * new_weight
        new_pdf[i-1] = new_pdf[i-1] * old_weight + new_prob * new_weight

    new_breaks = list(fspn.breaks)
    if len(left_most) != 0:
        new_breaks = [np.min(dataset) - EPSILON] + new_breaks
        new_pdf = [len(left_most)/new_card_actual * new_weight] + new_pdf

    if len(right_most) != 0:
        new_breaks = new_breaks + [np.max(dataset) + EPSILON]
        new_pdf = new_pdf + [len(right_most) / new_card_actual * new_weight]

    new_cdf = np.cumsum(new_pdf)
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"
    assert np.isclose(new_cdf[-1], 1), f"incorrect cdf, with max {new_cdf[-1]}"

    fspn.breaks = new_breaks
    fspn.pdf = new_pdf
    fspn.cdf = new_cdf
    """


def update_leaf_Multi_Histogram(fspn, dataset, ds_context):
    """
        Insert the new data into the original multi-histogram and update the parameter.
    """
    if fspn.range:
        if len(fspn.scope + fspn.condition) == dataset.shape[1]:
            idx = sorted(fspn.scope + fspn.condition)
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        elif len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]:
            idx = sorted(fspn.scope + list(fspn.range.keys()))
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        else:
            assert False

    shape = dataset.shape
    new_card = len(dataset)
    if new_card == 0:
        return
    dataset = dataset[~np.isnan(dataset)]
    dataset = dataset.reshape(shape)
    new_card_actual = len(dataset)  # the cardinality without nan.
    new_nan_perc = new_card_actual / new_card

    old_card = fspn.cardinality
    fspn.cardinality = old_card + new_card
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_weight = old_card / (new_card + old_card)
    new_weight = new_card / (new_card + old_card)
    fspn.nan_perc = old_weight * fspn.nan_perc + new_weight * new_nan_perc  # update nan_perc

    if new_card_actual == 0:
        return
    old_weight = old_card_actual / (new_card_actual + old_card_actual)
    new_weight = new_card_actual / (new_card_actual + old_card_actual)

    new_breaks_list = list(fspn.breaks)
    left_added = [False] * len(new_breaks_list)
    right_added = [False] * len(new_breaks_list)
    assert len(new_breaks_list) == dataset.shape[1], "mismatch number of breaks and data dimension"
    for i in range(len(new_breaks_list)):
        new_breaks = list(new_breaks_list[i])
        # new value out of bound of original breaks, adding new break
        if np.min(dataset[:, i]) < new_breaks[0]:
            new_breaks = [np.min(dataset[:, i]) - EPSILON] + new_breaks
            left_added[i] = True
        if np.max(dataset[:, i]) > new_breaks[-1]:
            new_breaks = new_breaks + [np.max(dataset[:, i]) + EPSILON]
            right_added[i] = True
        new_breaks_list[i] = np.asarray(new_breaks)

    new_pdf, new_breaks_list = np.histogramdd(dataset, bins=new_breaks_list)
    new_pdf = new_pdf / np.sum(new_pdf)
    old_pdf = np.zeros(new_pdf.shape)
    assert len(new_pdf.shape) == len(new_breaks_list)
    index = []
    for i in range(len(new_pdf.shape)):
        start = 0
        end = new_pdf.shape[i]
        if left_added[i]:
            start += 1
        if right_added[i]:
            end -= 1
        index.append(slice(start, end))
    old_pdf[tuple(index)] = fspn.pdf
    new_pdf = old_pdf * old_weight + new_pdf * new_weight
    new_cdf = multidim_cumsum(new_pdf)
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"

    fspn.breaks = new_breaks_list
    fspn.pdf = new_pdf
    fspn.cdf = new_cdf


def update_leaf_Merge(fspn, dataset, ds_context):
    """
    Insert the new data into the original merge leave and update the parameter.
    """
    if fspn.range is None:
        assert len(fspn.scope) == dataset.shape[1]
        idx_all = sorted(fspn.scope)
    else:
        assert len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]
        idx_all = sorted(fspn.scope + list(fspn.range.keys()))
    for leaf in fspn.leaves:
        if leaf.range is None:
            idx = [idx_all.index(i) for i in leaf.scope]
        else:
            idx = [idx_all.index(i) for i in sorted(leaf.scope + leaf.condition)]
        update_leaf(leaf, dataset[:, idx], ds_context)


def split_data_by_range(dataset, rect, scope):
    """
    split the new data by the range specified by a split node
    """
    local_data = copy.deepcopy(dataset)
    attrs = list(rect.keys())
    inds = sorted(scope + attrs)
    for attr in attrs:
        lrange = rect[attr]
        if type(lrange[0]) == tuple:
            left_bound = lrange[0][0]
            right_bound = lrange[0][1]
        elif len(lrange) == 1:
            left_bound = lrange[0]
            right_bound = lrange[0]
        else:
            left_bound = lrange[0]
            right_bound = lrange[1]
        i = inds.index(attr)
        indx = np.where((left_bound <= local_data[:, i]) & (local_data[:, i] <= right_bound))[0]
        local_data = local_data[indx]
    return local_data


def split_data_by_cluster_center(dataset, center, seed=17):
    """
    split the new data based on kmeans center
    """
    k = len(center)
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.cluster_centers_ = np.asarray(center)
    cluster = kmeans.predict(dataset)
    res = []
    for i in np.sort(np.unique(cluster)):
        local_data = dataset[cluster == i, :]
        res.append(local_data)
    return res


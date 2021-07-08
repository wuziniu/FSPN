import time
import copy
import numpy as np
from Learning.utils import convert_to_scope_domain
from sklearn.cluster import KMeans
try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time
from Learning.splitting.RDC import rdc_test
from Structure.nodes import Product, Sum, Factorize, Leaf
from Structure.leaves.fspn_leaves.Multi_Histograms import Multi_histogram, multidim_cumsum
from Structure.leaves.fspn_leaves.Histograms import Histogram
from Structure.leaves.fspn_leaves.Merge_leaves import Merge_leaves
from Inference.inference import EPSILON



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
    return rdc_adjacency_matrix, scope_loc, condition_loc


def top_down_update(fspn, ds_context, data_insert=None, data_delete=None):
    """
        Updates the FSPN when a new dataset arrives. The function recursively traverses the
        tree and inserts the different values of a dataset at the according places.
        At every sum node, the child node is selected, based on the minimal euclidian distance to the
        cluster_center of on of the child-nodes.
    """
    if fspn.range:
        if data_insert:
            assert data_insert.shape[1] == len(fspn.scope) + len(fspn.range), \
                f"mismatched data shape {data_insert.shape[1]} and {len(fspn.scope) + len(fspn.condition)}"
        if data_delete:
            assert data_delete.shape[1] == len(fspn.scope) + len(fspn.range), \
                f"mismatched data shape {data_delete.shape[1]} and {len(fspn.scope) + len(fspn.condition)}"
    else:
        if data_insert:
            assert data_insert.shape[1] == len(fspn.scope), \
                f"mismatched data shape {data_insert.shape[1]} and {len(fspn.scope)}"
        if data_delete:
            assert data_delete.shape[1] == len(fspn.scope), \
                f"mismatched data shape {data_delete.shape[1]} and {len(fspn.scope)}"

    if isinstance(fspn, Leaf):
        # TODO: indepence test along with original_dataset for multi-leaf nodes
        update_leaf(fspn, ds_context, data_insert, data_delete)
        return None

    elif isinstance(fspn, Factorize):
        left_cols = [fspn.scope.index(i) for i in fspn.children[0].scope]
        if data_insert:
            left_insert = data_insert[:, left_cols]
        else:
            left_insert = None
        if data_delete:
            left_delete = data_delete[:, left_cols]
        else:
            left_delete = None
        top_down_update(fspn.children[0], ds_context, left_insert, left_delete)
        top_down_update(fspn.children[1], ds_context, data_insert, data_delete)

    elif isinstance(fspn, Sum) and fspn.range is not None:
        # a split node
        assert fspn.cluster_centers == [], fspn
        for child in fspn.children:
            assert child.range is not None, child
            if data_insert:
                new_data_insert = split_data_by_range(data_insert, child.range, child.scope)
            else:
                new_data_insert = None
            if data_delete:
                new_data_delete = split_data_by_range(data_delete, child.range, child.scope)
            else:
                new_data_delete = None
            top_down_update(child, ds_context, new_data_insert, new_data_delete)

    elif isinstance(fspn, Sum):
        # a sum node
        assert len(fspn.cluster_centers) != 0, fspn
        origin_cardinality = fspn.cardinality
        if data_insert:
            new_data_insert = split_data_by_cluster_center(data_insert, fspn.cluster_centers)
            fspn.cardinality += len(data_insert)
        else:
            new_data_insert = [None]*len(fspn.children)
        if data_delete:
            new_data_delete = split_data_by_cluster_center(data_delete, fspn.cluster_centers)
            fspn.cardinality -= len(data_delete)
        else:
            new_data_delete = [None]*len(fspn.children)

        for i, child in enumerate(fspn.children):
            dl_insert = len(new_data_insert[i]) if new_data_insert[i] else 0
            dl_delete = len(new_data_delete[i]) if new_data_delete[i] else 0
            child_cardinality = origin_cardinality * fspn.weights[i]
            child_cardinality += dl_insert
            child_cardinality -= dl_delete
            fspn.weights[i] = child_cardinality / fspn.cardinality
            top_down_update(child, ds_context, new_data_insert[i], new_data_delete[i])


    elif isinstance(fspn, Product):
        # TODO: indepence test along with original_dataset
        for child in fspn.children:
            index = [fspn.scope.index(s) for s in child.scope]
            if data_insert:
                new_data_insert = data_insert[:, index]
            else:
                new_data_insert = None
            if data_delete:
                new_data_delete = data_delete[:, index]
            else:
                new_data_delete = None
            top_down_update(child, ds_context, new_data_insert, new_data_delete)


def update_leaf(fspn, ds_context, data_insert, data_delete):
    """
    update the parameter of leaf distribution, currently only support histogram.
    """
    if isinstance(fspn, Histogram):
        if data_insert:
            insert_leaf_Histogram(fspn, ds_context, data_insert)
        if data_delete:
            delete_leaf_Histogram(fspn, ds_context, data_delete)
    elif isinstance(fspn, Multi_histogram):
        if data_insert:
            insert_leaf_Multi_Histogram(fspn, ds_context, data_insert)
        if data_delete:
            delete_leaf_Multi_Histogram(fspn, ds_context, data_delete)
    elif isinstance(fspn, Merge_leaves):
        if data_insert:
            insert_leaf_Merge(fspn, ds_context, data_insert)
        if data_delete:
            delete_leaf_Merge(fspn, ds_context, data_delete)
    else:
        # TODO: implement the update of other leaf nodes
        assert False, "update of other node type is not yet implemented!!!!"


def insert_leaf_Histogram(fspn, ds_context, dataset):
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


def delete_leaf_Histogram(fspn, ds_context, dataset):
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

    old_card = fspn.cardinality
    fspn.cardinality = old_card - new_card
    assert fspn.cardinality >= 0, f"not enough data to delete"
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_card_nan = old_card * (1-fspn.nan_perc)
    new_card_nan = new_card - new_card_actual
    fspn.nan_perc = (old_card_nan-new_card_nan) / fspn.cardinality  # update nan_perc

    if new_card_actual == 0:
        return
    delete_weight = new_card_actual / old_card_actual
    remain_weight = 1 - delete_weight

    if np.min(dataset) < fspn.breaks[0] or np.max(dataset) > fspn.breaks[-1]:
        assert False, "deleted value out of bound of original breaks"

    delete_pdf, new_breaks = np.histogram(dataset, bins=fspn.breaks)
    delete_pdf = delete_pdf / np.sum(delete_pdf)
    old_pdf = fspn.pdf

    assert len(delete_pdf) == len(old_pdf) == len(new_breaks) - 1, "lengths mismatch"
    new_pdf = (old_pdf - delete_pdf * delete_weight) / remain_weight
    assert np.sum(new_pdf < 0) == 0, f"incorrect pdf, with negative entree {new_pdf[new_pdf < 0]}"
    new_cdf = np.zeros(len(new_pdf) + 1)
    for i in range(len(new_pdf)):
        if i == 0:
            new_cdf[i + 1] = new_pdf[i]
        else:
            new_cdf[i + 1] = new_pdf[i] + new_cdf[i]
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"
    assert np.isclose(new_cdf[-1], 1), f"incorrect cdf, with max {new_cdf[-1]}"

    fspn.pdf = new_pdf
    fspn.cdf = new_cdf

def insert_leaf_Multi_Histogram(fspn, ds_context, dataset):
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

def delete_leaf_Multi_Histogram(fspn, ds_context, dataset):
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

    old_card = fspn.cardinality
    fspn.cardinality = old_card - new_card
    assert fspn.cardinality >= 0, f"not enough data to delete"
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_card_nan = old_card * (1 - fspn.nan_perc)
    new_card_nan = new_card - new_card_actual
    fspn.nan_perc = (old_card_nan - new_card_nan) / fspn.cardinality  # update nan_perc

    if new_card_actual == 0:
        return
    delete_weight = new_card_actual / old_card_actual
    remain_weight = 1 - delete_weight

    breaks_list = list(fspn.breaks)
    assert len(breaks_list) == dataset.shape[1], "mismatch number of breaks and data dimension"
    for i in range(len(breaks_list)):
        new_breaks = list(breaks_list[i])
        if np.min(dataset[:, i]) < new_breaks[0] or np.max(dataset[:, i]) > new_breaks[-1]:
            assert False, "deleted value out of bound of original breaks"

    delete_pdf, breaks_list = np.histogramdd(dataset, bins=breaks_list)
    delete_pdf = delete_pdf / np.sum(delete_pdf)
    old_pdf = fspn.pdf
    assert delete_pdf.shape == old_pdf.shape
    new_pdf = (old_pdf - delete_pdf * delete_weight) / remain_weight
    assert np.sum(new_pdf < 0) == 0, f"incorrect pdf, with negative entree {new_pdf[new_pdf < 0]}"
    new_cdf = multidim_cumsum(new_pdf)
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"

    fspn.pdf = new_pdf
    fspn.cdf = new_cdf


def insert_leaf_Merge(fspn, ds_context, dataset):
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
        if isinstance(fspn, Histogram):
            insert_leaf_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Multi_histogram):
            insert_leaf_Multi_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Merge_leaves):
            insert_leaf_Merge(fspn, ds_context, dataset[:, idx])
        else:
            assert False, "Not implemented yet"


def delete_leaf_Merge(fspn, ds_context, dataset):
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
        if isinstance(fspn, Histogram):
            delete_leaf_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Multi_histogram):
            delete_leaf_Multi_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Merge_leaves):
            delete_leaf_Merge(fspn, ds_context, dataset[:, idx])
        else:
            assert False, "Not implemented yet"


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

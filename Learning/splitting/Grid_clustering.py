import numpy as np
import logging
from Inference.inference import EPSILON
from Learning.splitting.RDC import rdc_cca, rdc_transformer
from Learning.utils import convert_to_scope_domain
from Structure.StatisticalTypes import MetaType

logger = logging.getLogger(__name__)


def get_optimal_attribute(rdc_op, eval_func=np.max, fanout_attr=[]):
    """
    Using the pairwise rdc matrix to select the optimal attributes to split on
    :param rdc_op: the pair wise rdc value between attributes in data and scope attribute
    :param eval_func: choose between np.max, np.mean, np.median
    :param fanout_attr: the fanout indicator attributes
    :return: optimal attributes
    """
    (rdc_mat, scope_loc, condition_loc) = rdc_op
    if len(fanout_attr) == 0:
        query_attr_loc = condition_loc
    elif len(fanout_attr) == len(condition_loc):
        query_attr_loc = condition_loc
    else:
        query_attr_loc = [i for i in condition_loc if i not in fanout_attr]
    logger.debug(f"fanout_location {fanout_attr}, condition_location {condition_loc}, "
                 f"query_attr {query_attr_loc}")

    corr_min = 1.1
    opt_attr = None
    for i, c in enumerate(query_attr_loc):
        rdc_vals = np.ones(len(scope_loc))
        for j, s in enumerate(scope_loc):
            rdc_vals[j] = rdc_mat[c][s]
        corr = eval_func(rdc_vals)
        if corr < corr_min:
            corr_min = corr
            opt_attr = c
    return opt_attr, condition_loc.index(opt_attr)

def get_optimal_split_naive(data, attr, attr_type, n_clusters):
    """
    Split the attribute naively based on the median value
    :param data: local data containing only one attribute
    :param cond_fanout_data: The data containing fanout information on condition
    :param attr: optimal attribute to split on
    :param attr_type: meta_type of optimal attribute to split on
    :param n_clusters: number of clusters to split
    :param type:
    :return: the cluster id of each data point in the data
    """
    clusters = np.zeros(len(data))
    if attr_type == MetaType.BINARY:
        rect_range = []
        clusters[np.where(data == 0)] = 0
        rect_range.append({attr: [0]})
        clusters[np.where(data == 1)] = 1
        rect_range.append({attr: [1]})

    elif len(np.unique(data)) <= n_clusters:
        rect_range = []
        for i, uni in enumerate(list(np.unique(data))):
            clusters[np.where(data == uni)] = i
            rect_range.append({attr: [(uni, uni)]})
            
    elif n_clusters == 2:
        m = np.nanmedian(data)
        if m == np.nanmin(data):
            clusters[np.where(data <= m)] = 0
            clusters[np.where(data > m)] = 1
            rect_range = [{attr: [(np.nanmin(data), m)]}, {attr: [(np.nanmin(data[data > m]), np.nanmax(data))]}]
        else:
            clusters[np.where(data < m)] = 0
            clusters[np.where(data >= m)] = 1
            rect_range = [{attr: [(np.nanmin(data), np.nanmax(data[data < m]))]}, {attr: [(m, np.nanmax(data))]}]
            
    else:
        density, breaks = np.histogram(data[:, attr], bins=n_clusters)
        rect_range = []
        for i in range(len(density)):
            idxs = np.intersect1d(np.where(data[:, attr] >= breaks[i]), np.where(data[:, attr] < breaks[i+1]))
            clusters[idxs] = i
            rect_range.append({attr: [(breaks[i] + EPSILON, breaks[i+1])]})
    logger.info(f"find optimal clusters: {rect_range}")
    return clusters, rect_range


def sub_range_rdc_test(local_data, ds_context, scope, condition, attr_loc, rdc_sample=50000):
    if len(local_data) <= rdc_sample:
        data_sample = local_data
    else:
        data_sample = local_data[np.random.randint(local_data.shape[0], size=rdc_sample)]
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    meta_types = ds_context.get_meta_types_by_scope(scope_range)
    domains = ds_context.get_domains_by_scope(scope_range)
    rdc_features = rdc_transformer(
        data_sample, meta_types, domains, k=10, s=1.0 / 6.0, non_linearity=np.sin, return_matrix=False,
        rand_gen=None
    )
    print(data_sample.shape)
    print(rdc_features[0].shape)
    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=-1, max_nbytes=1024, backend="threading")(
        delayed(rdc_cca)((i, attr_loc, rdc_features)) for i in scope_loc
    )

    rdc_vector = np.zeros(len(scope))
    for i, rdc in zip(range(len(scope)), rdc_vals):
        rdc_vector[i] = rdc
    rdc_vector[np.isnan(rdc_vector)] = 0
    return rdc_vector

def get_equal_width_binning(local_data, num_bins):
    n = len(local_data)
    threshold = n/10000
    categories = sorted(list(np.unique(local_data)))
    bin_freq = 1 / num_bins
    print(bin_freq)
    freq = 0
    bins = []
    for i, k in enumerate(categories):
        freq += np.sum(local_data==k)/n
        if freq >= bin_freq:
            bins.append(k)
            freq = 0
        elif i==len(categories)-1:
            if freq*n > threshold:
                bins.append(k)
    return bins
    

def get_optimal_split(data, ds_context, scope, condition, attr_loc, attr, n_clusters=2,
                      rdc_sample=100000, eval_func=np.max, num_bins=15):
    """
    Split the attribute based on the pairwise RDC value but only support n_clusters=2
    :param data: local data containing only one attribute
    :param ds_context, scope, condition:
    :param attr_loc: the actual location of attr in the data
    :param attr: optimal attribute to split on
    :param n_clusters: number of clusters to split
    :param rdc_sample: number of samples to run rdc_test
    :param eval_func: evaluate function choose between np.max, np.mean and np.median
    :param num_bins: number of equal width bins we want to cut.
    :return: the cluster id of each data point in the data
    """
    assert n_clusters == 2, "only support dichotemy. If you want to split into multiple clusters, do it recursively"

    data_attr = data[:, attr_loc]
    clusters = np.zeros(len(data_attr))
    if len(np.unique(data_attr)) <= 2:
        rect_range = []
        for i, uni in enumerate(list(np.unique(data_attr))):
            clusters[np.where(data_attr == uni)] = i
            rect_range.append({attr: [(uni, uni + EPSILON)]})
    else:
        best_score = 2
        m = 0
        bins = get_equal_width_binning(data_attr, num_bins)
        print(bins)
        for i, k in enumerate(bins):
            print(f"checking value {k}")
            data_l = data[data_attr <= k]
            data_r = data[data_attr > k]
            print(len(data_l), len(data_r))
            if len(data_l) < 10 or len(data_r) < 10:
                continue
            rdc_vector_l = sub_range_rdc_test(data_l, ds_context, scope, condition, attr_loc, rdc_sample)
            score_l = eval_func(rdc_vector_l)
            rdc_vector_r = sub_range_rdc_test(data_r, ds_context, scope, condition, attr_loc, rdc_sample)
            score_r = eval_func(rdc_vector_r)
            if best_score > score_l+score_r:
                best_score = score_l+score_r
                m = k

        clusters[np.where(data_attr <= m)] = 0
        clusters[np.where(data_attr > m)] = 1
        rect_range = [{attr: [(np.nanmin(data_attr), m)]}, {attr: [(np.nanmin(data[data > m]), np.nanmax(data_attr))]}]
    logger.info(f"find optimal clusters: {rect_range}")
    return clusters, rect_range

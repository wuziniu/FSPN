from Learning.structureLearning import learn_structure
from Learning.structureLearning_binary import learn_structure_binary
from Learning.splitting.Condition_Clustering import *
from Learning.validity import is_valid
from Structure.nodes import Sum, assign_ids
from Structure.leaves.fspn_leaves.Multi_Histograms import create_multi_histogram_leaf
from Structure.leaves.fspn_leaves.Histograms import create_histogram_leaf
from Structure.leaves.binary.binary_leaf import create_binary_leaf
from Structure.leaves.binary.multi_binary_leaf import create_multi_binary_leaf
import itertools
import copy

import logging

logger = logging.getLogger(__name__)



def get_splitting_functions(cols, rows, ohe, threshold, rand_gen, n_jobs, max_sampling_threshold_rows=100000):
    from Learning.splitting.Clustering import get_split_rows_KMeans, get_split_rows_TSNE, get_split_rows_GMM
    if isinstance(cols, str):
        if cols == "rdc":
            from Learning.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs,
                                               max_sampling_threshold_cols=max_sampling_threshold_rows)
        elif cols == "poisson":
            from Learning.splitting.PoissonStabilityTest import get_split_cols_poisson_py
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
            split_rows_condition = None
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans(max_sampling_threshold_rows=max_sampling_threshold_rows)
            split_rows_condition = get_split_rows_condition_KMeans()
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
            split_rows_condition = get_split_rows_condition_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
            split_rows_condition = get_split_rows_condition_GMM()
        elif rows == "grid_naive":
            split_rows = get_split_rows_KMeans()
            split_rows_condition = get_split_rows_condition_Grid_naive()
        elif rows == "grid":
            split_rows = get_split_rows_KMeans(max_sampling_threshold_rows=max_sampling_threshold_rows)
            split_rows_condition = get_split_rows_condition_Grid()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows, split_rows_condition



def learn_FSPN(
    data,
    ds_context,
    cols="rdc",
    rows="grid_naive",
    threshold=0.3,
    rdc_sample_size=50000,
    rdc_strong_connection_threshold=0.75,
    multivariate_leaf=True,
    ohe=False,
    leaves=None,
    leaves_corr=None,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    if leaves is None:
        leaves = create_histogram_leaf

    if leaves_corr is None:
        leaves_corr = create_multi_histogram_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, cols, rows, threshold, ohe):
        split_cols, split_rows, split_rows_cond = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus,
                                                                          rdc_sample_size)

        return learn_structure(data, ds_context, split_rows, split_rows_cond, split_cols, leaves, leaves_corr,
                                  threshold=threshold, rdc_sample_size=rdc_sample_size, 
                                  rdc_strong_connection_threshold=rdc_strong_connection_threshold,
                                  multivariate_leaf=multivariate_leaf)

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, cols, rows, threshold, ohe)



def learn_FSPN_binary(
    data,
    ds_context,
    cols="rdc",
    rows="grid_naive",
    threshold=0.3,
    rdc_sample_size=50000,
    rdc_strong_connection_threshold=0.75,
    multivariate_leaf=True,
    ohe=False,
    leaves=None,
    leaves_corr=None,
    min_row_ratio=0.01,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    if leaves is None:
        leaves = create_binary_leaf

    if leaves_corr is None:
        leaves_corr = create_multi_binary_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, cols, rows, threshold, ohe):
        split_cols, split_rows, split_rows_cond = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus,
                                                                          rdc_sample_size)

        return learn_structure_binary(data, ds_context, split_rows, split_rows_cond, split_cols, leaves, leaves_corr,
                                  threshold=threshold, rdc_sample_size=rdc_sample_size,
                                  rdc_strong_connection_threshold=rdc_strong_connection_threshold,
                                  min_row_ratio=min_row_ratio, multivariate_leaf=multivariate_leaf)

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, cols, rows, threshold, ohe)


def evidence_query_generate(data, data_true, query_ncol_max=3):
    nrow, ncol = data.shape
    evidence_ncol = int(np.random.randint(2, ncol // 2))
    query_ncol = int(np.random.randint(query_ncol_max) + 1)
    evidence_col = np.random.choice(ncol, size=evidence_ncol, replace=False)
    left_col = [i for i in range(ncol) if i not in evidence_col]
    query_col = np.random.choice(left_col, size=query_ncol, replace=False)

    query_list = []
    for i in query_col:
        query_list.append(list(np.unique(data[:, i])))
    query_list = list(itertools.product(*query_list))

    query_left = np.zeros((len(query_list), ncol)) - np.infty
    query_right = np.zeros((len(query_list), ncol)) + np.infty

    ground_true = np.zeros(len(query_list))
    data_sub = copy.deepcopy(data_true)
    for i in evidence_col:
        idx = int(np.random.randint(nrow))
        val = data[idx, i]
        query_left[:, i] = val
        query_right[:, i] = val
        data_sub = data_sub[data_sub[:, i] == val]
    evidence_query = (copy.deepcopy(query_left[0]), copy.deepcopy(query_right[0]))

    for i, l in enumerate(query_list):
        s = None
        for j, pos in enumerate(query_col):
            query_left[i, pos] = l[j]
            query_right[i, pos] = l[j]
            if j == 0:
                s = (data_sub[:, pos] == l[j])
            else:
                s = s & (data_sub[:, pos] == l[j])
        ground_true[i] = len(np.where(s)[0]) / len(data_sub)

    return (query_left, query_right), evidence_query, ground_true


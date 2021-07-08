from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np

from Learning.splitting.Base import split_data_by_clusters, preproc
from Learning.splitting.Rect_approaximate import rect_approximate
from Learning.splitting.Grid_clustering import get_optimal_attribute, get_optimal_split_naive, get_optimal_split
import logging

logger = logging.getLogger(__name__)


def get_split_rows_condition_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    #only gathering on conditioned values
    def split_rows_KMeans(local_data, ds_context, scope, condition, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        range_idx = sorted(scope+condition)
        condition_idx = []
        for i in condition:
            condition_idx.append(range_idx.index(i))

        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data[:, condition_idx])

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans


def get_split_rows_condition_TSNE(n_clusters=2, pre_proc=None, ohe=False, seed=17, verbose=10, n_jobs=-1):
    # https://github.com/DmitryUlyanov/Multicore-TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import os

    ncpus = n_jobs
    if n_jobs < 1:
        ncpus = max(os.cpu_count() - 1, 1)

    def split_rows_KMeans(local_data, ds_context, scope, condition, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        range_idx = sorted(scope + condition)
        condition_idx = []
        for i in condition:
            condition_idx.append(range_idx.index(i))
        cond_data = data[:, condition_idx]

        kmeans_data = TSNE(n_components=3, verbose=verbose, n_jobs=ncpus, random_state=seed).fit_transform(cond_data)
        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(kmeans_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans


def get_split_rows_condition_DBScan(eps=2, min_samples=10, pre_proc=None, ohe=False):
    def split_rows_DBScan(local_data, ds_context, scope, condition, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        range_idx = sorted(scope + condition)
        condition_idx = []
        for i in condition:
            condition_idx.append(range_idx.index(i))

        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data[:, condition_idx])

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_DBScan


def get_split_rows_condition_GMM(n_clusters=2, pre_proc=None, ohe=False, seed=17, max_iter=100, n_init=2, covariance_type="full"):
    """
    covariance_type can be one of 'spherical', 'diag', 'tied', 'full'
    """

    def split_rows_GMM(local_data, ds_context, scope, condition, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        range_idx = sorted(scope + condition)
        condition_idx = []
        for i in condition:
            condition_idx.append(range_idx.index(i))

        estimator = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=seed,
        )

        clusters = estimator.fit(data[condition_idx, :]).predict(data[:, condition_idx])

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_GMM


def get_split_rows_condition_Grid_naive(n_clusters=2, pre_proc=None, ohe=False):
    # Using a grid based cluster (self invented naive version)
    def split_rows_Grid(local_data, ds_context, scope, condition, rdc_mat=None, cond_fanout_data=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        fanout_attr = [i for i in condition if i in ds_context.fanout_attr]
        idx_range = sorted(scope+condition)
        fanout_attr_loc = [idx_range.index(i) for i in fanout_attr]

        opt_attr, opt_attr_idx = get_optimal_attribute(rdc_mat, fanout_attr=fanout_attr_loc)
        logger.info(f"find optimal attribute: {condition[opt_attr_idx]}")
        clusters, range_slice = get_optimal_split_naive(data[:, opt_attr], condition[opt_attr_idx],
                                            ds_context.meta_types[condition[opt_attr_idx]], n_clusters)
        temp_res = split_data_by_clusters(local_data, clusters, scope, rows=True)
        if cond_fanout_data is not None:
            assert len(cond_fanout_data) == 2, "incorrect shape for conditional fanout data"
            assert len(cond_fanout_data[1]) == len(data), "mismatched data length"
            fanout_res = split_data_by_clusters(cond_fanout_data[1], clusters, scope, rows=True)
        res = []
        i = 0
        for data_slice, scope_slice, proportion in temp_res:
            if cond_fanout_data is not None:
                res.append((data_slice, range_slice[i], proportion, (cond_fanout_data[0], fanout_res[i][0])))
            else:
                res.append((data_slice, range_slice[i], proportion, None))
            i += 1
        return res

    return split_rows_Grid

def get_split_rows_condition_Grid(n_clusters=2, pre_proc=None, ohe=False, eval_func=np.max, seed=17):
    # Using a grid based cluster (self invented)
    def split_rows_Grid(local_data, ds_context, scope, condition, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        fanout_attr = [i for i in scope if i in ds_context.fanout_attr]
        idx_range = sorted(scope + condition)
        fanout_attr_loc = [idx_range.index(i) for i in fanout_attr]

        opt_attr, opt_attr_idx = get_optimal_attribute(rdc_mat, fanout_attr=fanout_attr_loc)
        clusters, range_slice = get_optimal_split(data, ds_context, scope, condition, opt_attr,
                                                  condition[opt_attr_idx], n_clusters, eval_func=eval_func)
        temp_res = split_data_by_clusters(local_data, clusters, scope, rows=True)
        res = []
        i = 0
        for data_slice, scope_slice, proportion in temp_res:
            res.append((data_slice, range_slice[i], proportion))
            i += 1
        return res

    return split_rows_Grid

def get_split_rows_condition_Rect(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    # Using a hyper-rectangle to approximate the learned cluster region of k-means (or other method)
    #To do: This is not implemented yet!!!!!
    def split_rows_Rect(local_data, ds_context, scope, condition, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        range_idx = sorted(scope + condition)
        condition_idx = []
        for i in condition:
            condition_idx.append(range_idx.index(i))
        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data[:, condition_idx])
        clusters, range_slice = rect_approximate(data[:, condition_idx], clusters)
        temp_res = split_data_by_clusters(local_data, clusters, scope, rows=True)
        res = []
        i = 0
        for data_slice, scope_slice, proportion in temp_res:
            res.append((data_slice, range_slice[i], proportion))
            i += 1
        return res

    return split_rows_Rect

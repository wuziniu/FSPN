from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
from Learning.splitting.Base import split_data_by_clusters, preproc
from Learning.splitting.Rect_approaximate import rect_approximate
from Learning.splitting.Grid_clustering import get_optimal_attribute, get_optimal_split
import logging

logger = logging.getLogger(__name__)


def get_split_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17, max_sampling_threshold_rows=200000):
    def split_rows_KMeans(local_data, ds_context, scope, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        if data.shape[0] > max_sampling_threshold_rows:
            data_sample = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]

            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit(data_sample).predict(data)
            center = kmeans.cluster_centers_
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit_predict(data)
            center = kmeans.cluster_centers_

        return split_data_by_clusters(local_data, clusters, scope, center, rows=True)

    return split_rows_KMeans


def get_split_rows_TSNE(n_clusters=2, pre_proc=None, ohe=False, seed=17, verbose=10, n_jobs=-1):
    # https://github.com/DmitryUlyanov/Multicore-TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import os

    ncpus = n_jobs
    if n_jobs < 1:
        ncpus = max(os.cpu_count() - 1, 1)

    def split_rows_KMeans(local_data, ds_context, scope, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        kmeans_data = TSNE(n_components=3, verbose=verbose, n_jobs=ncpus, random_state=seed).fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        clusters = kmeans.fit_predict(kmeans_data)
        center = kmeans.cluster_centers_

        return split_data_by_clusters(local_data, clusters, scope, center, rows=True)

    return split_rows_KMeans


def get_split_rows_DBScan(eps=2, min_samples=10, pre_proc=None, ohe=False):
    def split_rows_DBScan(local_data, ds_context, scope, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)
        center = dbscan.components_
        return split_data_by_clusters(local_data, clusters, scope, center, rows=True)

    return split_rows_DBScan



def get_split_rows_GMM(n_clusters=2, pre_proc=None, ohe=False, seed=17, max_iter=100, n_init=2, covariance_type="full"):
    """
    covariance_type can be one of 'spherical', 'diag', 'tied', 'full'
    """

    def split_rows_GMM(local_data, ds_context, scope, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        estimator = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=seed,
        )
        center = estimator.means_
        clusters = estimator.fit(data).predict(data)

        return split_data_by_clusters(local_data, clusters, scope, center, rows=True)

    return split_rows_GMM

def get_split_rows_Grid(n_clusters=2, pre_proc=None, ohe=False, seed=17, max_sampling_threshold_rows=200000):
    #Using a grid based cluster (self invented)
    def split_rows_Grid(local_data, ds_context, scope, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        opt_attr = get_optimal_attribute(local_data, rdc_mat)
        clusters = get_optimal_split(local_data, opt_attr, n_clusters)
        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_Grid


def get_split_rows_Rect(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    # Using a hyper-rectangle to approximate the learned cluster region of k-means (or other method)
    # Not Implemented yet!!!!
    def split_rows_Rect(local_data, ds_context, scope, rdc_mat=None):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)
        clusters = rect_approximate(local_data, clusters)
        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_Rect


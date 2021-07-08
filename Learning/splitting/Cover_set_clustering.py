import numpy as np
import logging
import math
from Inference.inference import EPSILON

logger = logging.getLogger(__name__)


def get_optimal_attribute(rdc_op):
    """
    Using the pairwise rdc matrix to select the optimal attributes to split on
    :param data: local data (conditional)
    :param rdc_op: the pair wise rdc value between attributes in data and scope attribute
    :return: optimal attributes
    """
    (rdc_mat, scope_loc, condition_loc) = rdc_op
    corr = np.ones(len(condition_loc))
    for i, c in enumerate(condition_loc):
        curr_max = 0
        for s in scope_loc:
            if rdc_mat[c][s] > curr_max:
                curr_max = rdc_mat[c][s]
        corr[i] = curr_max
    logger.info(f"find optimal attribute: {np.argmin(corr)}")
    opt_attr = np.argmin(corr)
    return condition_loc[opt_attr], opt_attr


def find_Cover_Set(score):
    """
    :param score: a matrix where H[i, j] recording the division value from part i to part j
    :return: a list of the points in the cover set
    """
    dim = score.shape[0]
    mask = np.ones(dim + 1)
    mask[0] = 0

    height = math.ceil(math.log(dim, 2))
    for p in range(1, height + 1):  # bottom-up scanning different levels
        for j in range(0, dim):  # level-wise scanning different nodes
            pos_start = j * pow(2, p) + 1
            pos_next = (j+1) * pow(2, p)
            if pos_next < dim or (pos_start < dim and pos_next >= dim):
                pos_end = min(pos_next, dim-1)
                pos_mid = math.ceil((pos_end + pos_start) / 2)

                if score[pos_start, pos_end] < score[pos_start, pos_mid] + score[pos_mid, pos_end]:
                    # update the division point
                    mask[pos_start: pos_end] = 0
                    mask[pos_end] = 1
                    print("node: [", pos_start, " : ", pos_end, "] --> [", pos_start, " : ", pos_mid, "] + [",
                          pos_mid,
                          " : ", pos_end, "]")
                    print("update the mask to ", mask[1:])

                else: # update the score
                    score[pos_start, pos_end] = score[pos_start, pos_mid] + score[pos_mid, pos_end]
                    print("node: [", pos_start, " : ", pos_end, "] --> [", pos_start, " : ", pos_mid, "] + [",
                          pos_mid,
                          " : ", pos_end, "]")
                    print("update the score to ")
                    print(score)

    cover_set = np.where(mask == 1)
    print(cover_set)
    return cover_set

def get_entropy_matrix(data):
    """
    create a matrix H where H[i, j] recording the entropy value from part i to part j
    :param data: a numpy arrow of shape (n,), for continous variables, it must be discretized first.
    :return: H
    """
    n = len(data)
    category = list(np.unique(data))
    k = len(category)
    H = np.zeros((k+1, k+1))
    probs = np.asarray([np.sum(data == i)/n for i in category])
    assert np.sum(probs) == 1, "probs doesn't sum to 1"
    for i in range(0, k+1):
        for j in range(i+1, k+1):
            interval_probs = probs[i:j]/np.sum(probs[i:j])
            H[i, j] = -np.sum(interval_probs * np.log2(interval_probs))


def get_optimal_split_CS(data, attr, n_clusters=2, sample_size=1000000):
    """
    Split the attribute naively based on the median value
    :param data: local data containing only one attribute
    :param attr: optimal attribute to split on
    :param n_clusters: number of clusters to split
    :param sample_size: number of sample to use to determine the splitting point
    :return: the cluster id of each data point in the data
    """
    clusters = np.zeros(len(data))
    if len(np.unique(data)) <= n_clusters:
        rect_range = []
        for i, uni in enumerate(list(np.unique(data))):
            clusters[np.where(data == uni)] = i
            rect_range.append({attr: [(uni, uni + EPSILON)]})
    else:
        if len(data) > sample_size:
            data_sample = data[np.random.randint(data.shape[0], size=sample_size), :]
            H = get_entropy_matrix(data_sample)
        else:
            H = get_entropy_matrix(data)
        cluster_points = find_Cover_Set(H, n_clusters)


    return clusters, rect_range


if __name__ == "__main__":
    dim = 8

    H = np.random.randint(0, 100, (dim + 1) * (dim + 1))
    H = np.reshape(H, [dim + 1, dim + 1])
    H[5, 5] = 0
    H[6, 6] = 0
    H[5, 7] = 888
    H[1, 8] = 999
    print(H[1:, 1:])

    find_Cover_Set(H)

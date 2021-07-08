import numpy as np
import pandas as pd


def discretize_series(data, n_mcv, n_bins, epsilon=0.01):
    """
    Map every value to category, binning the small categories if there are more than n_mcv categories.
    Map intervals to categories for efficient model learning
    return:
    s: discretized series
    n_distinct: number of distinct values in a mapped category (could be empty)
    encoding: encode the original value to new category (will be empty for continous attribute)
    mapping: map the new category to pd.Interval (for continuous attribute only)
    """
    n = len(data)
    (uniques, value_counts) = np.unique(data, return_counts=True)
    sort_idx = np.argsort(value_counts)

    # Treat most common values
    mcv_domain = []
    mcv_nums = []
    for i in sort_idx[:n_mcv]:
        mcv_domain.append(uniques[i])
        mcv_nums.append(value_counts[i])

    # Treat least common values
    lcv_n = np.sum(value_counts[sort_idx[n_mcv:]])
    lcv_domain = []
    lcv_nums = []
    cum_count = 0
    bin_width = lcv_n / n_bins
    for i, uni in enumerate(uniques):
        if uni not in mcv_domain:
            if cum_count == 0:
                left = uni
            cum_count += value_counts[i]
            if cum_count >= bin_width:
                lcv_domain.append(left)
                lcv_nums.append(cum_count)
                cum_count = 0
    if cum_count != 0:
        lcv_domain.append(left)
        lcv_nums.append(cum_count)

    domain_breaks = np.asarray(mcv_domain+lcv_domain)
    pdfs = np.asarray(mcv_nums+lcv_nums)/n
    sort_idx = np.argsort(domain_breaks)
    domain_breaks = domain_breaks[sort_idx]
    pdfs = pdfs[sort_idx]
    breaks = []
    for i, c in enumerate(domain_breaks):
        if i != 0:
            breaks.append(domain_breaks[i - 1] + (c - domain_breaks[i - 1]) / 2)
        else:
            breaks.append(c - epsilon)
    if uniques[-1] in mcv_domain:
        breaks.append(domain_breaks[-1] + epsilon)
    else:
        breaks.append(uniques[-1] + epsilon)
    return pdfs, breaks

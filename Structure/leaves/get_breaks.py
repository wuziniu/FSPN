import numpy as np
from Inference.inference import EPSILON

def get_breaks(data, domain):
    # Get the reasonable breaks for 1D histogram
    if len(domain) > 2:
        domain_breaks = []
        for i, c in enumerate(domain):
            if i != 0:
                domain_breaks.append(domain[i - 1] + (c - domain[i - 1]) / 2)
            else:
                domain_breaks.append(c - EPSILON)
        domain_breaks.append(domain[-1] + EPSILON)
        unique = list(np.sort(np.unique(data)))
        new_breaks = []
        for i, c in enumerate(domain):
            if c in unique:
                if len(new_breaks) == 0:
                    new_breaks.append(domain_breaks[i])
                    new_breaks.append(domain_breaks[i + 1])
                else:
                    if new_breaks[-1] == domain_breaks[i]:
                        new_breaks.append(domain_breaks[i + 1])
                    else:
                        new_breaks.append(domain_breaks[i])
                        new_breaks.append(domain_breaks[i + 1])
        return np.asarray(new_breaks)

    else:
        breaks = []
        domain = np.sort(np.unique(data))
        for i, c in enumerate(domain):
            if i != 0:
                breaks.append(domain[i - 1] + (c - domain[i - 1]) / 2)
            else:
                breaks.append(c - EPSILON)
        breaks.append(domain[-1] + EPSILON)
        return np.asarray(breaks)

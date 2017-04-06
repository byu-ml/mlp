import numpy as np


def get_indices(X, multi_sets, k_down, k_up, shuffle=True):
    if not multi_sets:
        idx = [(0, i) for i in np.arange(k_down, len(X[0]) - k_up)]
        if shuffle:
            np.random.shuffle(idx)
        return idx
    else:
        indices = [[(i, j) for j in np.arange(k_down, len(X[i]) - k_up)] for i in np.arange(len(X))]
        idx = []
        for x in indices:
            idx.extend(x)
        if shuffle:
            np.random.shuffle(idx)
        return idx

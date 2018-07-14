"""Calculations for gait metrics."""

import numpy as np
from scipy.spatial.distance import cdist

import modules.iterable_funcs as itf


def stride_lengths(df_stance):

    series = df_stance.groupby('number')['position'].apply(np.stack)

    return [np.median(cdist(a, b)) for a, b in itf.pairwise(series)]

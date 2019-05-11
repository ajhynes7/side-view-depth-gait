"""Constants used by multiple scripts."""

import numpy as np


PART_TYPES = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']

PART_CONNECTIONS = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [3, 5], [1, 3]])

import math


def normalize_array(x):
    """
    x : array_like
        Array of length 2 or greater
    """

    max_value = np.nanmax(x)
    min_value = np.nanmin(x)

    return (x - min_value) / (max_value - min_value)


def sigmoid(x, a=1):
    """
    Sigmoid function (produces the sigmoid curve).
    
    Parameters
    ----------
    x : {[type]}
        [description]
    a : {number}, optional
        [description] (the default is 1, which [default_description])
    
    Returns
    -------
    number
        [description]
    """

    return 1 / (1 + math.exp(-a * x))


def root_mean_square(x):

    return np.sqrt(sum(x**2) / x.size)


def centre_of_mass(points, masses):

    _, n_dimensions = points.shape

    total = np.zeros(n_dimensions)

    for i, point in enumerate(points):
        mass = masses[i]
        total += mass * point

    centre = total / sum(masses)
    return centre.reshape(-1, 1)


def gaussian(x, mu, sigma):

    coeff = 1.0 / np.sqrt(np.pi * sigma**2)
    exponent = np.exp(- (x - mu)**2 / (2 * sigma**2))

    return coeff * exponent
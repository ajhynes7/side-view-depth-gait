"""Functions related to classes."""


def get_properties(class_object):
    """
    Calculate all properties of a class instance.

    Parameters
    ----------
    class_object : object
        Instance of class.

    Returns
    -------
    property_dict : dict
        Dictionary containing all properties of class instance.
        Dict keys are property names, dict values are property values.

    Notes
    -----
    Example usage:

    get_properties(steve)
    {'name': 'Steve', 'age': 8, 'weight': 30}

    """
    property_dict = {}

    class_name = class_object.__class__

    for var in vars(class_name):

        if isinstance(getattr(class_name, var), property):

            property_dict[var] = getattr(class_object, var)

    return property_dict

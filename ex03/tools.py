import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array. x can be a one-dimensional (m * 1) or two-dimensional (m * n) array.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    col = np.array([1 for _ in range(len(x))])
    if (len(x.shape) == 1):
        res = np.empty((x.shape[0], 2))
        for i in range(len(x)):
            res[i] = [1, x[i]]
    elif (len(x.shape) == 2):
        res = np.empty((x.shape[0], x.shape[1] + 1))
        for i in range(x.shape[0]):
            res[i] = np.concatenate(([1], x[i]))
    return res
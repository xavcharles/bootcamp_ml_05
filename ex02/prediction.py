import numpy as np

def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a one-dimensional array of size m.
    theta: has to be an numpy.ndarray, a one-dimensional array of size 2.
    Returns:
    y_hat as a numpy.ndarray, a one-dimensional array of size m.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    result = np.array([theta[0] + theta[1] * x[i] for i in range(len(x))])
    return result

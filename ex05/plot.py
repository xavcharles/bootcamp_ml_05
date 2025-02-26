import numpy as np
import matplotlib.pyplot as plt

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

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a one-dimensional array of size m.
    theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a two-dimensional array of shape m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    y = add_intercept(x)
    return np.dot(y, theta)

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a one-dimensional array of size m.
    y: has to be an numpy.array, a one-dimensional array of size m.
    theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    plt.plot(x, y, label="points only", color='blue', linestyle="None", marker="o", markersize=4)
    plt.plot(x, predict_(x, theta), label="prediction line", color='orange', linestyle='-', marker='None')
    # plt.grid(True)
    # plt.legend()
    plt.show()
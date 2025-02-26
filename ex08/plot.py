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

# def loss_elem_(y, y_hat):
#     """
#     Description:
#     Calculates all the elements (y_pred - y)^2 of the loss function.
#     Args:
#     y: has to be an numpy.array, a two-dimensional array of shape m * 1.
#     y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
#     Returns:
#     J_elem: numpy.array, a array of dimension (number of the training examples, 1).
#     None if there is a dimension matching problem.
#     None if any argument is not of the expected type.
#     Raises:
#     This function should not raise any Exception.
#     """
#     J_elem = np.array([pow(y_hat[i] - y[i], 2) for i in range(y.shape[0])])
#     return J_elem

def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a one-dimensional array of size m.
    y_hat: has to be an numpy.array, a one-dimensional array of size m.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """
    return (np.dot(y_hat - y, y_hat - y) / (2*y.shape[0]))

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, one-dimensional array of size m.
    y: has to be an numpy.ndarray, one-dimensional array of size m.
    theta: has to be an numpy.ndarray, one-dimensional array of size 2.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    plt.plot(x, y, label="points only", color='blue', linestyle="None", marker="o", markersize=4)
    plt.plot(x, y_hat, label="prediction line", color='orange', linewidth=1, linestyle='-', marker='None')
    plt.vlines(x=x, ymin=y, ymax=y_hat, color='red', linewidth=2, linestyle='--', label='Vertical Line')
    plt.title(f"Cost : {loss_(y, y_hat)}")
    # plt.grid(True)
    # plt.legend()
    plt.show()
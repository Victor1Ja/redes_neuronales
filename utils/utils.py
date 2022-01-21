import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

def sigmoid(z):
    """
    Return:
    s -- sigmoid(z)
    """
    
    
    s = 1./(1+np.exp(-z))
    
    return s

def RELU(z):
    """
    Return:
    s -- RELU(z)
    """
    s = np.maximum(z,0)
    return s
def Parametric_RELU(z, a):
    """
    z -- vector
    a -- parameter int
    Return:
    s -- Parametric_RELU(z)
    """
    s = np.maximum(z, z*a )
    return s

def Leaky_RELU(z):
    """
    Return:
    s -- RELU(z)
    """
    # s = np.where( z>0 , z, 0.01*z)
    s = np.maximum(z, z*0.01 )
    return s

def Tanh(z):
    """
    Return:
    s -- tanh(z)
    """
    s = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

    return s

def Silu(z):
    """
    Return:
    s -- sigmoid(z)*z
    """
    s = z*sigmoid(z)

    return s



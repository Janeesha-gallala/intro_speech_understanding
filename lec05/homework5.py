import numpy as np

def center_of_gravity(x):
    """
    Find the center of gravity of a vector, x.
    If x=[x0,x1,...,xn], then you should return
    c = ( 0*x0 + 1*x1 + 2*x2 + ... + n*xn ) / sum(x)
    where n = len(x)-1.
    """
    x = np.array(x)
    indices = np.arange(len(x))
    c = np.dot(indices, x) / np.sum(x)
    return c


def matched_identity(x):
    """
    Create an identity matrix that has the same number of rows as x has elements.
    """
    n = len(x)
    I = np.eye(n)
    return I


def sine_and_cosine(t_start, t_end, t_steps):
    """
    Create a time axis, and compute its cosine and sine.
    """
    t = np.linspace(t_start, t_end, t_steps)
    x = np.cos(t)
    y = np.sin(t)
    return t, x, y

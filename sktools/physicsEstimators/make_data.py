import numpy as np
import pandas as pd
import numpy as np
from scipy.constants import G
import random

random.seed(0)


def newton_equation(m1=1, m2=1, r=1, G=G):
    """
    Newton´s Equation
    :param m1: Mass first body in kg
    :param m2: Mass second body in kg
    :param r: Distance between m1 and m2 in meters
    :param G: Gravitational constant
    :return: Newtons equation in the Internation System of Units
    """
    return G * m1 * m2 * (1 / r ** 2)


def movement_equation(x0=1, v=1, a=1, t=1):
    return x0 + v*t + 0.5 * a * t*t


def make_newton(
    samples=100,
    m1_min=0.001,
    m1_max=1,
    m2_min=0.001,
    m2_max=1,
    r_min=1,
    r_max=10,
    G=G,
    dataframe=True,
):
    """
    Creates a sample of data of the Newton Equation
    :param samples: Number of samples
    :param m1_min: Minimum value in the range of the mass of the first body
    :param m1_max: Maximum value in the range of the mass of the first body
    :param m2_min: Minimum value in the range of the mass of the second body
    :param m2_max: Maximum value in the range of the mass of the second body
    :param r_min: Minimum value of the distance between the bodies
    :param r_max: Maximum value of the distance between the bodies
    :param dataframe: wether it returns a dataframe or an array
    :return: data sample
    """
    data = []
    for n in range(samples):
        m1 = (m1_max - m1_min) * np.random.random() + m1_min
        m2 = (m2_max - m2_min) * np.random.random() + m2_min
        r = (r_max - r_min) * np.random.random() + r_min

        data.append([m1, m2, r, newton_equation(m1=m1, m2=m2, r=r, G=G)])
    if dataframe:
        return pd.DataFrame(data=data, columns=["m1", "m2", "r", "f"])
    else:
        return data


def make_movement_data(
    samples=100,
    x0_min=1,
    x0_max=2,
    v_min=1,
    v_max=2,
    a_min=1,
    a_max=2,
    t_min=1,
    t_max=2,
    dataframe=True,
):
    '''
    Cinematic movement equation

    :param samples: Number of samples
    :param x0_min: Minimum value for x_0
    :param x0_max: Maximum value for x_0
    :param v_min: Minimum value for v
    :param v_max: Maximum value for v
    :param a_min: Minimum value for a
    :param a_max: Maximum value for a
    :param t_min: Minimum value for t
    :param t_max: Maximum value for t
    :param dataframe: either dataframe or not
    :return:
    '''
    data = []
    for n in range(samples):
        x0 = (x0_max - x0_min) * np.random.random() + x0_min
        v = (v_max - v_min) * np.random.random() + v_min
        a = (a_max - a_min) * np.random.random() + a_min
        t = (t_max - t_min) * np.random.random() + t_min

        data.append([x0, v, a, t, movement_equation(x0=x0, v=v, a=a, t=t)])
    if dataframe:
        return pd.DataFrame(data=data, columns=["x0", "v", "a", "t", "pos"])
    else:
        return data


import numpy as np
from pybnn.util import AttrDict

r"""
Sources:

Infinity GO 1D - http://infinity77.net/global_optimization/test_functions_1d.html
"""


class ObjectiveFunction1D(object):
    r"""
    Base Class for defining arbitrary objective function containers. Each objective function object should define the
    following parameters at the time of initialization:
    domain: a 2-tuple of values, the recommended input domain for this objective function.
    minimum: a 2-tuple of values, one known global minima of this objective function for the given range, given as the
        tuple (argmin, min) where argmin is the argument x that minimizes f(x) and min is the actual minimum value.
    func: a callable object, which is passed the input x everytime the objective function object itself is to be
        evluated on a given input.
    name: the name of the objective function, to be returned by __str__ calls.
    """

    def __init__(self, domain=(0.0, 1.0), minima=(0.0, 0.0), func=lambda x: x, name="obj"):
        self.domain = domain
        self.minima = minima
        self.func = func
        self.name = name


    def __call__(self, x):
        return self.func(x)


    def __str__(self):
        return str(self.name)


#############################################################
#####################  NON PARAMETERISED  ###################
#############################################################

nonParameterisedObjectiveFunctions = AttrDict()

def inf_go_prob_2(x):
    r"""Infinity GO 1D Problem 02"""
    return np.sin(x) + np.sin(10./3.0 * x)

nonParameterisedObjectiveFunctions.infinityGO2 = ObjectiveFunction1D(
    domain=(2.7, 7.5),
    minima=(5.145735, -1.899599),
    func=inf_go_prob_2,
    name="Infinity GO Problem 02"
)

def inf_go_prob_4(x):
    r"""Infinity GO 1D Problem 04"""
    return -(16 * x ** 2 - 24 * x + 5) * np.exp(-x)

nonParameterisedObjectiveFunctions.infinityGO4 = ObjectiveFunction1D(
    domain=(1.9, 3.9),
    minima=(2.868034, -3.85045),
    func=inf_go_prob_4,
    name="Infinity GO Problem 04"
)

def inf_go_prob_5(x):
    r"""Infinity GO 1D Problem 05"""
    return -(1.4 - 3 * x) * np.sin(18 * x)

nonParameterisedObjectiveFunctions.infinityGO5 = ObjectiveFunction1D(
    domain=(0.0, 1.2),
    minima=(0.96609, -1.48907),
    func=inf_go_prob_5,
    name="Infinity GO Problem 05"
)

def inf_go_prob_6(x):
    r"""Infinity GO 1D Problem 06"""
    return -(x + np.sin(x)) * np.exp(-x ** 2)

nonParameterisedObjectiveFunctions.infinityGO6 = ObjectiveFunction1D(
    domain=(-10.0, 10.0),
    minima=(0.67956, -0.824239),
    func=inf_go_prob_6,
    name="Infinity GO Problem 06"
)

def inf_go_prob_7(x):
    r"""Infinity GO 1D Problem 07"""
    return np.sin(x) + np.sin(10. / 3. * x) + np.log(x) - 0.84 * x + 3

nonParameterisedObjectiveFunctions.infinityGO7 = ObjectiveFunction1D(
    domain=(2.7, 7.5),
    minima=(5.19978, -1.6013),
    func=inf_go_prob_7,
    name="Infinity GO Problem 07"
)

def inf_go_prob_9(x):
    r"""Infinity GO 1D Problem 09"""
    return np.sin(x) + np.sin(2./3. * x)

nonParameterisedObjectiveFunctions.infinityGO9 = ObjectiveFunction1D(
    domain=(3.1, 20.4),
    minima=(17.039, -1.90596),
    func=inf_go_prob_9,
    name="Infinity GO Problem 09"
)

def inf_go_prob_10(x):
    r"""Infinity GO 1D Problem 10"""
    return -x * np.sin(x)

nonParameterisedObjectiveFunctions.infinityGO10 = ObjectiveFunction1D(
    domain=(0., 10.),
    minima=(7.9787, -7.916727),
    func=inf_go_prob_10,
    name="Infinity GO Problem 010"
)

def inf_go_prob_11(x):
    r"""Infinity GO 1D Problem 11"""
    return 2 * np.cos(x) + np.cos(2 * x)

nonParameterisedObjectiveFunctions.infinityGO11 = ObjectiveFunction1D(
    domain=(-np.pi / 2., 2. * np.pi),
    minima=(2.09439, -1.5),
    func=inf_go_prob_11,
    name="Infinity GO Problem 11"
)

def inf_go_prob_14(x):
    r"""Infinity GO 1D Problem 014"""
    return -np.exp(-x) * np.sin(2 * np.pi * x)

nonParameterisedObjectiveFunctions.infinityGO14 = ObjectiveFunction1D(
    domain=(0., 4.),
    minima=(0.224885, -0.788685),
    func=inf_go_prob_14,
    name="Infinity GO Problem 014"
)

def inf_go_prob_15(x):
    r"""Infinity GO 1D Problem 015"""
    return (x ** 2 - 5 * x + 6) / (x ** 2 + 1)

nonParameterisedObjectiveFunctions.infinityGO15 = ObjectiveFunction1D(
    domain=(-5., 5.),
    minima=(2.41422, -0.03553),
    func=inf_go_prob_15,
    name="Infinity GO Problem 015"
)


#############################################################
#######################  PARAMETERISED  #####################
#############################################################

parameterisedObjectiveFunctions = AttrDict()


def sin_func(x, w=1.0, t=0.0):
    r"""Sin(wx+t), where w and t are floating point values."""
    return np.sin(w * x + t)

parameterisedObjectiveFunctions.sin = ObjectiveFunction1D(
    domain=(-np.pi, np.pi),
    minima=(-np.pi / 2., -1.0),
    func=sin_func,
    name="Sine"
)


def cos_func(x, w=1.0, t=0.0):
    r"""Cos(wx+t), where w and t are floating point values."""
    return np.cos(w * x + t)

parameterisedObjectiveFunctions.cos = ObjectiveFunction1D(
    domain=(-np.pi, np.pi),
    minima=(-np.pi, -1.0),
    func=cos_func,
    name="Cosine"
)


def sinc_func(x, w=1.0, t=0.0):
    r"""Normalized sinc(wx+t), where w and t are floating point values."""
    return np.sinc(w * x + t)

parameterisedObjectiveFunctions.sinc = ObjectiveFunction1D(
    domain=(-5., 5.),
    minima=(1.430299, -0.217234),
    func=sinc_func,
    name="Normalized Sinc"
)
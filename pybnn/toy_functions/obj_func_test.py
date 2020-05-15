from pybnn.toy_functions import parameterisedObjectiveFunctions, nonParameterisedObjectiveFunctions
from pybnn.toy_functions.toy_1d import ObjectiveFunction1D
import matplotlib.pyplot as plt
# import matplotlib.axes.Axes
import numpy as np

step_size = 1e-3

for func_name, func in nonParameterisedObjectiveFunctions.items():
# for func_name, func in parameterisedObjectiveFunctions.items():
    func: ObjectiveFunction1D
    xlim = func.domain
    fig, ax = plt.subplots(1, 1, squeeze=True)
    # ax: matplotlib.axes.Axes
    ax.set_xlim(left=xlim[0], right=xlim[1])
    x = np.arange(xlim[0], xlim[1], step_size)
    fx = func(x)
    xmin = x[np.argmin(fx)]
    fmin = np.min(fx)
    ax.plot(x, fx)
    ax.set_title(func.name)
    ax.grid()
    ax.scatter(xmin, fmin, c='red', marker='o')
    ax.scatter(func.minima[0], func.minima[1], c='black', marker='x')
    print(f"Displaying plot for {func_name}")
    plt.show()
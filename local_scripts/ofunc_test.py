import sys
sys.path.append('/home/archit/master_project/pybnn')
from pybnn.util.universal_utils import simple_plotter as plotter
from sklearn.model_selection import train_test_split
from pybnn.toy_functions.toy_1d import nonParameterisedObjectiveFunctions as ofuncs
import numpy as np
import matplotlib.pyplot as plt
obj = ofuncs.infinityGO2
lower, upper = obj.domain
X = np.arange(lower, upper, 0.05)
y = obj(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True)


env_width = np.full_like(ytest, fill_value=0.1 * (np.max(ytest) - np.min(ytest)))

pred = np.stack((xtest, ytest, env_width), axis=1)
train = np.stack((xtrain, ytrain), axis=1)

np.random.shuffle(pred)

print(pred.shape)
print(train.shape)

fig = plotter(pred=pred, train=train, plot_variances=True)
plt.show()

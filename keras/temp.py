import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import pandas
import math

from sklearn.metrics import mean_squared_error

print('%.2f RMSE' % (math.sqrt(mean_squared_error([1,1,1,1], [2,2,2,2]))))
print('%.2f RMSE' % (math.sqrt(mean_squared_error([1,1], [1,2]))))


plt.plot([0,1,4,9,16])
plt.savefig("temp.png")

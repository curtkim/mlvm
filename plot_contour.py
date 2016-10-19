import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

print x.shape
print y.shape
xx, yy = np.meshgrid(x, y, sparse=True)
print xx.shape
print yy.shape

z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)

plt.savefig('contour.png')
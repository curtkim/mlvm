import numpy as np

X = np.array([
  [0,0],
  [0,1],
  [1,0],
  [1,1],
], dtype=np.int)

W = np.array([
  [1,1],
  [1,1],
], dtype=np.int)

c = np.array([
  [0,-1]
], dtype=np.int)

Y = X.dot(W) + c

# rectified linear unit or ReLU
Y[0][1] = 0

w = np.array([
  [1],
  [-2]
], dtype=np.int)

print Y.dot(w)

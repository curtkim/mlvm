import numpy
import theano.tensor as T
from theano import function

x = T.dmatrix('x')
y = T.dmatrix('y')
#z = x + y
#z = T.concatenate([x, y])
z = T.dot(x, y)
f = function([x, y], z)

print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
import theano, theano.tensor as T
import numpy as np

from losses import tukey_biweight


a = np.arange(16).reshape((2, 2, 2, 2)) + 1
b = np.arange(16).reshape((2, 2, 2, 2)) + 1

# Invalid data
b[0,0,0,1] = 0

a = a.astype(np.float32)
b = b.astype(np.float32)

x = T.tensor4("x")
y = T.tensor4("y")

z = tukey_biweight(x, y)

f = theano.function([x, y], z,on_unused_input='warn')

print "MSE is: " + str(((a-b)**2).mean())

print "Tukey-Loss is: " + str(f(np.log(a),b))
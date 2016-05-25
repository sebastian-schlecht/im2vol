import theano, theano.tensor as T
import numpy as np

from losses import tukey_biweight, spatial_gradient

plot = True

a = np.ones(10000).reshape((10, 10, 10, 10))
b = np.arange(10000).reshape((10, 10, 10, 10))


a = a.astype(np.float32)
b = b.astype(np.float32)

x = T.tensor4("x")
y = T.tensor4("y")

z = spatial_gradient(x, y)

f = theano.function([x, y], z,on_unused_input='warn')

print "MSE is: " + str(((a-b)**2).mean())

r = f(np.log(a),b)
print "Custom-Loss is: " + str(r)

if(plot):
    import matplotlib.pyplot as plt
    r = r.squeeze()
    plt.plot(r)
plt.show()
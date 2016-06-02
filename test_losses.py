import theano, theano.tensor as T
import numpy as np

from losses import tukey_biweight, spatial_gradient, _tukey_cost

plot = True

a = np.ones(100000) / 10000.
b = (np.arange(100000) - 50000.) / 10000.


a = a.astype(np.float32)
b = b.astype(np.float32)

x = T.vector("x")
y = T.vector("y")

z = _tukey_cost(x - y)

f = theano.function([x, y], z,on_unused_input='warn')

print "MSE is: " + str(((a-b)**2).mean())

r = f(a, b)
print "Custom-Loss is: " + str(r)

if(plot):
    import matplotlib.pyplot as plt
    r = r.squeeze()
    plt.plot(r)
plt.show()
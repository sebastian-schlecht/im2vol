import theano, theano.tensor as T
import numpy as np

from losses import tukey_biweight

plot = True

a = np.ones(100).reshape((1, -1))
b = np.arange(100).reshape((1, -1))


a = a.astype(np.float32)
b = b.astype(np.float32)

x = T.matrix("x")
y = T.matrix("y")

z = tukey_biweight(x, y)

f = theano.function([x, y], z,on_unused_input='warn')

print "MSE is: " + str(((a-b)**2).mean())

r = f(np.log(a),b)
print "Tukey-Loss is: " + str(r)

if(plot):
    import matplotlib.pyplot as plt
    r = r.squeeze()
    plt.plot(r)
plt.show()
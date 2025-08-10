%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()

def synthetic_data(w, b, num_examples):

    X = np.random.normal (0, 1, (num_examples, len(w)))
    y =np.dot(X,w) + b 
    y += np.random.normal(0, 0.01, y.shape)
    return X, y
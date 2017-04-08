import theano.tensor as tensor
def elu(x):

    return tensor.switch(x > 0, x, tensor.expm1(x))

elu(2)
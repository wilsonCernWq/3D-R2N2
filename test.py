import theano.tensor as tensor
def elu(x):
    """Exponential Linear Unit :math:`\\varphi(x) = (x > 0) ? x : e^x - 1`
    The Exponential Linear Unit (ELU) was introduced in [1]_. Compared to the
    linear rectifier :func:`rectify`, it has a mean activation closer to zero
    and nonzero gradient for negative input, which can help convergence.
    Compared to the leaky rectifier :class:`LeakyRectify`, it saturates for
    highly negative inputs.
    Parameters
    ----------
    x : float32
    """
    return tensor.switch(x > 0, x, tensor.expm1(x))

elu(2)

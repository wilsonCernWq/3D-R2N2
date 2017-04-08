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
        The activation (the summed, weighed input of a neuron).
    Returns
    -------
    float32
        The output of the exponential linear unit for the activation.
    Notes
    -----
    In [1]_, an additional parameter :math:`\\alpha` controls the (negative)
    saturation value for negative inputs, but is set to 1 for all experiments.
    It is omitted here.
    References
    ----------
    .. [1] Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015):
       Fast and Accurate Deep Network Learning by Exponential Linear Units
       (ELUs), http://arxiv.org/abs/1511.07289
    """
    return tensor.switch(x > 0, x, tensor.expm1(x))

elu(2)
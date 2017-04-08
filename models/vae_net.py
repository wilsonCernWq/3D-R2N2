import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import Net, tensor5
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, \
    EltwiseMultiplyLayer, get_trainable_params

class VAEnet(Net):

    def network_definition(self):
        self.x = tensor5()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h
        n_vox = self.n_vox

        n_deconvfilter = [128, 128, 128, 64, 32, 2]

        s_all = s_update[0]
        s_last = s_all[-1]
        gru_s = InputLayer(s_shape, s_last)
        unpool7 = Unpool3DLayer(gru_s)
        conv7 = Conv3DLayer(unpool7, (n_deconvfilter[1], 30, 30, 30))
        rect7 = LeakyReLU(conv7)
        unpool8 = Unpool3DLayer(rect7)
        conv8 = Conv3DLayer(unpool8, (n_deconvfilter[2], 15, 15, 15))
        rect8 = LeakyReLU(conv8)
        unpool9 = Unpool3DLayer(rect8)
        conv9 = Conv3DLayer(unpool9, (n_deconvfilter[3], 13, 13, 13))
        rect9 = LeakyReLU(conv9)
        # unpool10 = Unpool3DLayer(rect9)
        conv10 = Conv3DLayer(rect9, (n_deconvfilter[4], 7, 7, 7))
        rect10 = LeakyReLU(conv10)


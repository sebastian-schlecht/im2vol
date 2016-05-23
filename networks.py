import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import batch_norm
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import FlattenLayer, ReshapeLayer
from lasagne.nonlinearities import rectify
from lasagne.layers import TransposedConv2DLayer as DeconvLayer, ConcatLayer


def vnet(input_var=None):
    # Input
    l_in = InputLayer(shape=(None, 3, 240, 320), input_var=input_var)

    # Contractive Stack 1
    l_conv_0_1 = ConvLayer(l_in, num_filters=64, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_0_2 = ConvLayer(l_conv_0_1, num_filters=64, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_pool_0_2 = PoolLayer(l_conv_0_2, pool_size=(2, 2))

    # Contractive Stack 2
    l_conv_1_1 = ConvLayer(l_pool_0_2, num_filters=128, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_1_2 = ConvLayer(l_conv_1_1, num_filters=128, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_pool_1_2 = PoolLayer(l_conv_1_2, pool_size=(2, 2))

    # Contractive Stack 3
    l_conv_2_1 = ConvLayer(l_pool_1_2, num_filters=256, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_2_2 = ConvLayer(l_conv_2_1, num_filters=256, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_pool_2_2 = PoolLayer(l_conv_2_2, pool_size=(2, 2))

    # Contractive Stack 4
    l_conv_3_1 = ConvLayer(l_pool_2_2, num_filters=512, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_3_2 = ConvLayer(l_conv_3_1, num_filters=512, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_pool_3_2 = PoolLayer(l_conv_3_2, pool_size=(2, 2))

    # Bottleneck
    l_flat = FlattenLayer(l_pool_3_2)
    l_fc_1 = DenseLayer(l_flat, 4096)
    l_fc_2 = DenseLayer(l_fc_1, 19200)
    l_rs = ReshapeLayer(l_fc_2, (-1, 64, 15, 20))

    # Upscale Stack 1
    l_up_4_1 = DeconvLayer(l_rs, num_filters=64, filter_size=(2, 2), stride=(2, 2), crop=0, nonlinearity=None)
    l_concat_4_1 = ConcatLayer([l_up_4_1, l_conv_3_2])
    l_conv_4_1 = ConvLayer(l_concat_4_1, num_filters=512, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_4_2 = ConvLayer(l_conv_4_1, num_filters=512, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # Upscale Stack 2
    l_up_5_1 = DeconvLayer(l_conv_4_2, num_filters=256, filter_size=(2, 2), stride=(2, 2), crop=0, nonlinearity=None)
    l_concat_5_1 = ConcatLayer([l_up_5_1, l_conv_2_2])
    l_conv_5_1 = ConvLayer(l_concat_5_1, num_filters=256, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_5_2 = ConvLayer(l_conv_5_1, num_filters=256, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # Upscale Stack 3
    l_up_6_1 = DeconvLayer(l_conv_5_2, num_filters=128, filter_size=(2, 2), stride=(2, 2), crop=0, nonlinearity=None)
    l_concat_6_1 = ConcatLayer([l_up_6_1, l_conv_1_2])
    l_conv_6_1 = ConvLayer(l_concat_6_1, num_filters=128, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_6_2 = ConvLayer(l_conv_6_1, num_filters=128, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # Upscale Stack 4
    l_up_7_1 = DeconvLayer(l_conv_6_2, num_filters=64, filter_size=(2, 2), stride=(2, 2), crop=0, nonlinearity=None)
    l_concat_7_1 = ConcatLayer([l_up_7_1, l_conv_0_2])
    l_conv_7_1 = ConvLayer(l_concat_7_1, num_filters=64, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    l_conv_7_2 = ConvLayer(l_conv_7_1, num_filters=64, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad=1, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # Final prediction
    l_conv_8_1 = ConvLayer(l_conv_7_2, num_filters=1, filter_size=(1, 1), stride=(1, 1), pad=0)

    return l_conv_8_1




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


def residual_vnet(input_var=None, n=5):

    # Residual block from https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)

        return block


    # Residual upscaling block
    def residual_block_upscale(l, decrease_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if decrease_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters / 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if decrease_dim:
            if True:
                # projection shortcut, as option B in paper
                projection = batch_norm(DeconvLayer(l, num_filters=out_num_filters, filter_size=(2,2), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
            # TODO Use crop instead of pad an re-enable
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X.repeat(2, axis=2).repeat(2, axis=3), lambda s: (s[0], s[1], s[2]*2, s[3]*2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)

        return block

    # Input
    l_in = InputLayer(shape=(None, 3, 240, 320), input_var=input_var)

    # First layer
    l_s_1 = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 64 x h x w (320x240)
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 128 x h/2 x w/2 (160x120)
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 256 x h/4 x w/4 (80x60)
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # fourth stack of residual blocks, output is 512 x h/8 x w/8 (40x30)
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # fifth stack of residual blocks, output is 1024 x h/16 x w/16 (20x15)
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # Upsample Stack 1 output is 512 x h/8 x w/8 (40x30)
    l = residual_block_upscale(l, descrease_dim=True)
    for _ in range(1,n):
        l = residual_block_upscale(l)
    # Upsample Stack 2 output is 256 x h/4 x w/4 (80x60)
    l = residual_block_upscale(l, descrease_dim=True)
    for _ in range(1,n):
        l = residual_block_upscale(l)
    # Upsample Stack 3 output is 128 x h/2 x w/2 (160x120)
    l = residual_block_upscale(l, descrease_dim=True)
    for _ in range(1,n):
        l = residual_block_upscale(4)
    # Upsample Stack 1 output is 64 x h x w (320x240)
    l = residual_block_upscale(l, descrease_dim=True)
    for _ in range(1,n):
        l = residual_block_upscale(l)

    # Final prediction
    l = ConvLayer(l, num_filters=1, filter_size=(1, 1), stride=(1, 1), pad=0)

    return l






import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer, Upscale2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import batch_norm
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer, GlobalPoolLayer
from lasagne.layers import FlattenLayer, ReshapeLayer
from lasagne.nonlinearities import rectify
from lasagne.layers import TransposedConv2DLayer as DeconvLayer, ConcatLayer


def debug_net(input_var=None, depth=3):
    """
    Debug network which is small & fast
    :param input_var: Input variable
    :param depth: Depth of the net's core
    :return: lasagne.layer
    """
    # Input
    l_in = InputLayer(shape=(None, 3, 240, 320), input_var=input_var)
    l = l_in
    for _ in range(depth):
        l = batch_norm(ConvLayer(l, num_filters=64, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad="same",
                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    l = ConvLayer(l, num_filters=1, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad="same",
                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    return PoolLayer(l,pool_size=(2,2))


def residual_unet(input_var=None, n=3, un=1, connectivity=0):
    """
    Parametrized residual unet in lasagne for depth prediction
    Output dims are half the input dimensions
    :param input_var: Input variable
    :param n: Number of dimension residual convolutions per feature map resolution (contractive path)
    :param un: Number of residual convolutions for the upstream path (min 1)
    :param connectivity: Degree to which the branches are connected.
    :return: lasagne.layers
    """
    assert n > 0
    assert un > 0

    # residual upsampling according to Iro
    def residual_block_up(l, decrease_dim=False, projection=True, pad=True):
        input_num_filters = l.output_shape[1]
        
        if decrease_dim:
            out_num_filters = input_num_filters / 2
            # Upsample
            l = Upscale2DLayer(l, 2)
        else:
            out_num_filters = input_num_filters
        
        
         # Our switch to "cheat" our inital dimensions back
        if pad:
            padding = "same"
            proj_filter = (1,1)
            conv_filter = (3,3)
        else:
            padding = 0
            # Odd filters here but works to get the dimensions right
            conv_filter = (3,1)
            proj_filter = (3,1)
        
        
        # Now we can use a simple "normal" residual block
        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=conv_filter, stride=(1,1), nonlinearity=rectify, pad=padding, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if decrease_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=proj_filter, stride=(1,1), nonlinearity=None, pad=padding, b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            ## NOT IMPLEMENTED
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

        return block

    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        """
        Helper for a residual block using bottleneck architecture
        :param l: lasagne.layer Input layer
        :param increase_dim: boolean Increase outfilters and decrease spatial feature map sizes?
        :param projection: Use projection?
        :param pad: Padding
        :return:
        """
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        bottleneck_filters = out_num_filters // 4
        # We use a bottleneck approach here!
        stack_1 = batch_norm(
            ConvLayer(l, num_filters=bottleneck_filters, filter_size=(1, 1), stride=first_stride, nonlinearity=rectify,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
            ConvLayer(stack_1, num_filters=bottleneck_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_3 = batch_norm(
            ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1), nonlinearity=None,
                      pad='same', W=lasagne.init.HeNormal(), flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2), nonlinearity=None,
                              pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_3, projection]), nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_3, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_3, l]), nonlinearity=rectify)

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 240, 320), input_var=input_var)

    # First batch normalized layer
    first_kernel = 64
    l = batch_norm(
        ConvLayer(l_in, num_filters=first_kernel, filter_size=(5, 5), stride=(2, 2), nonlinearity=rectify, pad=2,
                  W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    print l.output_shape
    # First residual block
    for _ in range(n):
        l = residual_block(l)
    # Save reference before downsampling
    l_1 = l

    # second stack of residual blocks, output is 128x60x80
    l = residual_block(l, increase_dim=True)
    for _ in range(1, n):
        l = residual_block(l)

    # Save reference
    l_2 = l

    # third stack of residual blocks, output is 256x30x40
    l = residual_block(l, increase_dim=True)
    for _ in range(1, n):
        l = residual_block(l)
    # Save reference
    l_3 = l

    # forth stack of residual blocks, output is 512x15x20
    l = residual_block(l, increase_dim=True)
    for _ in range(1, n):
        l = residual_block(l)
    # Save reference
    l_4 = l
    # fifth stack of residual blocks, output is 1024x8x10
    l = residual_block(l, increase_dim=True, projection=True)
    for _ in range(1, n):
        l = residual_block(l)
    l_5 = l

    # Expansive path

    # first expansive block. seventh stack of residuals, output is 512x16x20
    l = residual_block_up(l, decrease_dim=True)
    for _ in range(1, un):
        l = residual_block(l)
    l_7 = l

    # We have to cheat here in order to get our initial feature map dimensions back
    # What we do is taking uneven filter dimensions to artificially generate the desired filter sizes
    # Question: Does this make sense or would cropping the first and last pixel row work better?!
    # first expansive block. seventh stack of residuals, output is 256x30x40
    if connectivity > 0:
        l = ConcatLayer([l_7, l_4])
    l = residual_block_up(l, decrease_dim=True, pad=False)
    for _ in range(1, un):
        l = residual_block(l)
    l_8 = l
    if connectivity > 1:
        l = ConcatLayer([l_8, l_3])
    # residual block #8, output is 128x60x80
    l = residual_block_up(l, decrease_dim=True)
    for _ in range(1, un):
        l = residual_block(l)
    l_9 = l
    if connectivity > 2:
        l = ConcatLayer([l_9, l_2])
    # residual block #9, output is 64x120x160
    l = residual_block_up(l, decrease_dim=True)
    for _ in range(1, n):
        l = residual_block(l)
    l_10 = l
    if connectivity > 3:
        l = ConcatLayer([l_10, l_1])

    # final convolution
    l = ConvLayer(l, num_filters=1, filter_size=(1, 1), stride=(1, 1), nonlinearity=rectify, pad="same",
                             W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    return l



def residual_unet_old(input_var = None, n=3,nu=3):
    # residual upsampling according to Iro
    def residual_block_up(l, decrease_dim=False, projection=True, pad=True):
        input_num_filters = l.output_shape[1]
        
        if decrease_dim:
            out_num_filters = input_num_filters / 2
            # Upsample
            l = Upscale2DLayer(l, 2)
        else:
            out_num_filters = input_num_filters
        
        
         # Our switch to "cheat" our inital dimensions back
        if pad:
            padding = "same"
            proj_filter = (1,1)
            conv_filter = (3,3)
        else:
            padding = 0
            # Odd filters here but works to get the dimensions right
            conv_filter = (3,1)
            proj_filter = (3,1)
        
        
        # Now we can use a simple "normal" residual block
        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=conv_filter, stride=(1,1), nonlinearity=rectify, pad=padding, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if decrease_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=proj_filter, stride=(1,1), nonlinearity=None, pad=padding, b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            ## NOT IMPLEMENTED
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

        return block
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False, pad=True):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters
            
       
        bottleneck = out_num_filters // 4
        stack_1 = batch_norm(ConvLayer(l, num_filters=bottleneck, filter_size=(1,1), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=bottleneck, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_3 = batch_norm(ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))


        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_3, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_3, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_3, l]),nonlinearity=rectify)

        return block
    
     # Building the network
    l_in = InputLayer(shape=(None, 3, 240, 320), input_var=input_var)
    
    # First batch normalized layer
    l = batch_norm(ConvLayer(l_in, num_filters=256, filter_size=(5,5), stride=(2,2), nonlinearity=rectify, pad=2, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    l = PoolLayer(l, pool_size=(2,2))
    # Save reference before downsampling
    l_1 = l
    
    # First residual block
    for _ in range(1,n):
        l = residual_block(l)    
    
    # Save reference
    l_2 = l
    
    # third stack of residual blocks, output is 512x30x40
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    # Save reference
    l_3 = l
    
    # forth stack of residual blocks, output is 1024x15x20
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    # Save reference
    l_4 = l
    # fifth stack of residual blocks, output is 2048x8x10
    l = residual_block(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block(l)
    l_5 = l
    
    # Expansive path
    
    # first expansive block. seventh stack of residuals, output is 256x16x20
    l = residual_block_up(l, decrease_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    l_6 = l
    
    # We have to cheat here in order to get our initial feature map dimensions back
    # What we do is taking uneven filter dimensions to artificially generate the desired filter sizes
    # Question: Does this make sense or would cropping the first and last pixel row work better?!
    # first expansive block. seventh stack of residuals, output is 256x30x40
    l = residual_block_up(l, decrease_dim=True, pad=False)
    for _ in range(1,nu):
        l = residual_block(l)
    l_7 = l
    
    # residual block #8, output is 128x60x80
    l = residual_block_up(l, decrease_dim=True)
    for _ in range(1,nu):
        l = residual_block(l)
    l_8 = l
    
    # residual block #9, output is 64x120x160
    l = residual_block_up(l, decrease_dim=True)
    for _ in range(1,nu):
        l = residual_block(l)
    l_9 = l
    
    
    # final convolution
    l = batch_norm(ConvLayer(l, num_filters=1, filter_size=(1,1), stride=(1,1), nonlinearity=rectify, pad="same", W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    return l

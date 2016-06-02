import numpy as np
import lasagne
from lasagne.layers import Pool2DLayer
import theano
import theano.tensor as T
import h5py
import time

DATASET = "./data/nyu_depth_combined_vnet2"
name = "resunet"
#model = "./data/rs_stack_1_spatial_grad_epoch_250.npz"
learn_stack = 0
model = None

from networks import vnet, d_rs_stack_1, d_rs_stack_2, residual_unet
from losses import scale_invariant_error, tukey_biweight, spatial_gradient, mse


def iterate_minibatches(inputs, targets, batchsize, shuffle=True, augment=True, downsample=1):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        # Random crops
        h = 240
        w = 320

        cy = np.random.randint(inputs.shape[2] - h, size=1)
        cx = np.random.randint(inputs.shape[3] - w, size=1)

        input_cropped = inputs[excerpt, :, cy:cy+h, cx:cx+w].copy()
        target_cropped = targets[excerpt, cy:cy+h:downsample, cx:cx+w:downsample].copy()

        yield input_cropped, target_cropped


def load_data():
    f = h5py.File(DATASET + ".hdf5")

    x_train = np.array(f["images"]).astype(np.float32)
    y_train = np.array(f["depths"]).astype(np.float32)

    # Subtract mean already and normalize std
    m = np.load(DATASET + ".npy").astype(np.float32)
    for i in range(len(x_train)):
        x_train[i] = (x_train[i] - m) / 68.

    return (x_train, y_train)


def main(num_epochs=200, lr=0.01, batch_size=16):
    # loss_func = spatial_gradient
    loss_func = spatial_gradient
    print "Building network"
    input_var = T.tensor4('inputs')
    target_var = T.tensor3('targets')
    # Reshape to enable usage in loss function
    target_reshaped = target_var.dimshuffle((0, "x", 1, 2))
    
    if learn_stack == 1 or learn_stack == 0:
        
        network = residual_unet(input_var=input_var)
        prediction = lasagne.layers.get_output(network)

        # Spatial grad / biweight / scale invariant error
        loss = loss_func(prediction, target_reshaped)
        # Add some L2
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        cost = loss + l2_penalty

        params = lasagne.layers.get_all_params(network, trainable=True)
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=sh_lr, momentum=0.9)
        # Load model weights
        if model is not None:
            print "Loading model weights %s" % model
            with np.load(model) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)
        
    
    elif learn_stack == 2:
        # Create stack 1
        in_, stack_1 = d_rs_stack_1(input_var)
        params_1 = lasagne.layers.get_all_params(stack_1, trainable=True)
        layers_1 = lasagne.layers.get_all_layers(stack_1)
        
        
        # Create stack 2
        stack_2 = d_rs_stack_2(input_layer=in_, prev=stack_1)
        params_2 = lasagne.layers.get_all_params(stack_2, trainable=True)
        layers_2 = lasagne.layers.get_all_layers(stack_2)
        
        layers_top = [l for l in layers_2 if l not in layers_1]
        
        # Fix BN gamma and beta for stack 1
        #_ = lasagne.layers.get_output(stack_1, deterministic=True)
        # Get output
        prediction = lasagne.layers.get_output(stack_2)
        
        # Compute loss
        loss = loss_func(prediction, target_reshaped)
        l2_penalty = lasagne.regularization.regularize_layer_params(layers_top, lasagne.regularization.l2) * 0.0001
        cost = loss + l2_penalty
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        # Only collect stack 2 weights for updates
        params = [param for param in params_2 if param not in params_1]
        
        updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=sh_lr, momentum=0.9)
        network = stack_2
        # Load model weights
        if model is not None:
            with np.load(model) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(layers_1, param_values)
        
    
    
    
    print "Compiling network"
    # We want to have the main loss only, not l2 values
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print "Loading data"
    X_train, Y_train = load_data()

    print "Starting training"
    train_losses = []
    for epoch in range(num_epochs):
        if epoch == num_epochs // 2:
            print "Decreasing LR"
            sh_lr.set_value(lr / 10)
        # shuffle training data
        train_indices = np.arange(X_train.shape[0])
        np.random.shuffle(train_indices)
        X_train = X_train[train_indices, :, :, :]
        Y_train = Y_train[train_indices, :, :]

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        bt = 0
        bidx = 0
        print "Training Epoch %i" % (epoch + 1)
        
        ds = 1
        if learn_stack == 1:
            ds = 4
        elif learn_stack == 2:
            ds = 2
        for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True, augment=True, downsample=ds):
            inputs, targets = batch 
            bts = time.time()
            err = train_fn(inputs, targets)
            bte = time.time()
            bt += (bte - bts)
            bidx += 1
            if bidx == 20 and epoch == 0:
                tpb = bt / bidx
                print "Average time per forward/backward pass: " + str(tpb)
                eta = time.time() + num_epochs * (tpb * (len(X_train)/batch_size))
                localtime = time.asctime( time.localtime(eta) )
                print "ETA: ", localtime
            
            train_losses.append(err)
            train_err += err
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    # Save
    np.savez('./data/' + name +'_epoch_%i.npz' % num_epochs, *lasagne.layers.get_all_param_values(network))
    np.save('./data/' + name + '_epoch_' + str(num_epochs) + '_loss.npy', np.array(train_losses))


if __name__ == '__main__':
    main()

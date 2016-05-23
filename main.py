import numpy as np
import lasagne
import theano
import theano.tensor as T
import h5py
import time

DATASET = "./data/nyu_depth_combined_vnet2"

from networks import build_vnet
from losses import scale_invariant_error


def iterate_minibatches(inputs, targets, batchsize, shuffle=True, augment=True):
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

        input_cropped = inputs[excerpt, :, cy:cy+h, cx:cx+w]
        target_cropped = targets[excerpt, cy:cy+h, cx:cx+w]

        yield input_cropped, target_cropped


def load_data():
    f = h5py.File(DATASET + ".hdf5")

    x_train = np.array(f["images"])
    y_train = np.array(f["depths"])

    # Subtract mean already and normalize std
    m = np.load(DATASET + ".npy")
    for i in range(len(x_train)):
        x_train[i] = (x_train[i] - m) / 68.

    return (x_train, y_train)


def main(num_epochs=10, lr=0.01, batch_size=4):
    print "Building network"
    input_var = T.tensor4('inputs')
    target_var = T.tensor3('targets')

    # Reshape to enable usage in loss function
    target_reshaped = target_var.reshape((-1, 1, 240, 320))

    network = build_vnet(input_var=input_var)
    prediction = lasagne.layers.get_output(network)

    # Scale invariant loss
    loss = scale_invariant_error(predictions=prediction, targets=target_reshaped)

    # Add some L2
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss = loss + l2_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)

    print "Compiling network"
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print "Loading data"
    X_train, Y_train = load_data()

    print "Starting training"
    for epoch in range(num_epochs):
        # shuffle training data
        train_indices = np.arange(X_train.shape[0])
        np.random.shuffle(train_indices)
        X_train = X_train[train_indices,:,:,:]
        Y_train = Y_train[train_indices,:,:]

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True, augment=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            print train_err / train_batches
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


if __name__ == '__main__':
    main()

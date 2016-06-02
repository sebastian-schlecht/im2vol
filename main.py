import numpy as np
import lasagne
from threading import Thread
import Queue
import theano
import theano.tensor as T
import h5py
import time

DATASET_TRAIN = "./data/nyu_depth_combined_vnet2"
DATASET_VAL = "./data/nyu_depth_combined_vnet2"
name = "resunet"
model = None
synced_prefetching = True


q = Queue.Queue(maxsize=20)

from networks import  residual_unet
from losses import spatial_gradient


def prefetch_proc(bs):
    """
    Open handle to DB and prefetch data
    :return:
    """
    f = h5py.File(DATASET_TRAIN + ".hdf5")
    length = f["images"].shape[0]
    perm = np.random.permutation(length)
    idx = 0
    # Subtract mean already and normalize std
    m = np.load(DATASET_TRAIN + ".npy").astype(np.float32)
    while True:
        pp = perm[idx:idx+bs]
        pp = np.sort(pp).tolist()
        images = np.array(f["images"][pp])
        labels = np.array(f["depths"][pp])

        for i in range(len(images)):
            images[i] = (images[i] - m) / 68.
        q.put((images, labels), block=True)

        if idx + bs > length:
            idx = 0
            perm = np.random.permutation(length)
        else:
            idx += bs


def start_prefetching_thread(batchsize):
    thread = Thread(target=prefetch_proc, args=(batchsize,))
    thread.daemon = True
    thread.start()
    return thread


def iterate_minibatches_synchronized(inputlen, targetlen, batchsize, shuffle=True, augment=True, downsample=1):
    """
    Iterate minibatches from a synced queue
    :param inputlen:
    :param targetlen:
    :param shuffle:
    :param augment:
    :param downsample:
    :return:
    """
    assert inputlen == targetlen
    for start_idx in range(0, inputlen - batchsize + 1, batchsize):
        # Random crops
        h = 240
        w = 320
        inputs, targets = q.get(block=True)
        cy = np.random.randint(inputs.shape[2] - h, size=1)
        cx = np.random.randint(inputs.shape[3] - w, size=1)

        input_cropped = inputs[:, :, cy:cy+h, cx:cx+w].copy()
        target_cropped = targets[:, cy:cy+h:downsample, cx:cx+w:downsample].copy()

        yield input_cropped, target_cropped


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


def load_val_data():
    f = h5py.File(DATASET_VAL + ".hdf5")

    x_train = np.array(f["images"]).astype(np.float32)
    y_train = np.array(f["depths"]).astype(np.float32)

    # Subtract mean already and normalize std
    m = np.load(DATASET_VAL + ".npy").astype(np.float32)
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
        
    print "Compiling network"
    # We want to have the main loss only, not l2 values
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [prediction, loss])
    f = h5py.File(DATASET_TRAIN + ".hdf5")
    length = f["images"].shape[0]
    f.close()
    print "Starting data prefetcher"
    start_prefetching_thread(batch_size)
    print "Loading validation data"
    x_val, y_val = load_val_data()
    print "Starting training"
    train_losses = []
    for epoch in range(num_epochs):
        if epoch == num_epochs // 2:
            print "Decreasing LR"
            sh_lr.set_value(lr / 10)
        # shuffle training data
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        bt = 0
        bidx = 0
        print "Training Epoch %i" % (epoch + 1)

        # Downsample factor
        ds = 1
        # Train for one epoch
        for batch in iterate_minibatches_synchronized(length, length, batch_size, shuffle=True, augment=True, downsample=ds):
            inputs, targets = batch 
            bts = time.time()
            err = train_fn(inputs, targets)
            bte = time.time()
            bt += (bte - bts)
            bidx += 1
            if bidx == 20 and epoch == 0:
                tpb = bt / bidx
                print "Average time per forward/backward pass: " + str(tpb)
                eta = time.time() + num_epochs * (tpb * (length/batch_size))
                localtime = time.asctime( time.localtime(eta) )
                print "ETA: ", localtime
            
            train_losses.append(err)
            train_err += err
            train_batches += 1


        # Validate model
        v_losses = []
        v_mse = []
        for i in range(x_val):
            x_in = x_val[i,:,9:9+240:ds, 12:12+320:ds]
            y_in = y_val[i,9:9+240:ds, 12:12+320:ds]
            v_pred, v_loss = val_fn(x_in, y_in)
            current_se = (v_pred - y_val[i]) ** 2
            v_mse.append(current_se)
            v_losses.append(v_losses)
        val_mse = np.array(v_mse).mean()
        val_loss = np.array(v_losses).mean()
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  val loss:\t\t{:.6f}".format(val_loss))
        print("  val mse:\t\t{:.6f}".format(val_mse))

    # Save
    np.savez('./data/' + name +'_epoch_%i.npz' % num_epochs, *lasagne.layers.get_all_param_values(network))
    np.save('./data/' + name + '_epoch_' + str(num_epochs) + '_loss.npy', np.array(train_losses))


if __name__ == '__main__':
    main()

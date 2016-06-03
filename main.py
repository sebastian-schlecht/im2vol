import numpy as np
import lasagne
from threading import Thread
import Queue
import theano
import theano.tensor as T
import h5py
import time, math

from networks import  residual_unet, debug_net, d_rs_stack_1
from losses import spatial_gradient, berhu

# Our shuffled dataset. Important that we store it in contigous blocks, and not chunks to have const. speed on variable batchsizes
DATASET_TRAIN = "/home/sebastianschlecht/depth_data/nyu_v1_shuffled"

# Validation dataset that we keep in memory
DATASET_VAL = "/home/sebastianschlecht/depth_data/nyu_depth_v2_resized"
DATASET_TRAIN = DATASET_VAL
# Name used for saving losses and models
name = "resunet"
# Init with model params
model = None
# Synced mode for prefetching. Should stay True, otherwise this script will break
synced_prefetching = True
# Threadsafe queue for prefetching
q = Queue.Queue(maxsize=20)

validate = False


def prefetch_proc(bs):
    """
    Open handle to DB and prefetch data. Also do random online augmention
    :return: None - Loops infinitely
    """
    f = h5py.File(DATASET_TRAIN + ".hdf5")
    length = f["images"].shape[0]
    idx = 0
    # Subtract mean already and normalize std
    m = np.load(DATASET_TRAIN + ".npy").astype(np.float32)
    while True:
        upper = min(idx+bs, length)
        images = np.array(f["images"][idx:upper]).copy()
        labels = np.array(f["depths"][idx:upper]).copy()
                
        for i in range(len(images)):
            images[i] = (images[i] - m) / 70.933385161726605
            # Flip horizontally with probability 0.5
            p = np.random.randint(2)
            if p > 0:
                images[i] = images[i, :, :, ::-1]
                labels[i] = labels[i, :, ::-1]

            # RGB we mult with a random value between 0.8 and 1.2
            r = np.random.randint(80,121) / 100.
            g = np.random.randint(80,121) / 100.
            b = np.random.randint(80,121) / 100.
            images[i, 0] = images[i, 0] * r
            images[i, 1] = images[i, 1] * g
            images[i, 2] = images[i, 2] * b
                

        q.put((images, labels), block=True)

        if idx + bs > length:
            idx = 0
        else:
            pass
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

        while True:
            try:
                inputs, targets = q.get(block=True,timeout=0.05)
                break
            except Queue.Empty:
                print "Prefetch Queue Empty"
        cy = np.random.randint(inputs.shape[2] - h, size=1)
        cx = np.random.randint(inputs.shape[3] - w, size=1)
        input_cropped = inputs[:, :, cy:cy+h, cx:cx+w]
        target_cropped = targets[:, cy:cy+h:downsample, cx:cx+w:downsample]
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
        x_train[i] = (x_train[i] - m) / 70.933385161726605
    f.close()
    return (x_train, y_train)


def main(num_epochs=5, lr=0.001, batch_size=16):
    loss_func = spatial_gradient
    # loss_func = berhu
    print "Building network"
    input_var = T.tensor4('inputs')
    target_var = T.tensor3('targets')
    # Reshape to enable usage in loss function
    target_reshaped = target_var.dimshuffle((0, "x", 1, 2))
    
    network = d_rs_stack_1(input_var=input_var)
    # Downsample factor
    ds = 4
    
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
        
    print "Compiling training model"
    # We want to have the main loss only, not l2 values
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    if validate:
        print "Compiling validation model"
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        val_loss = loss_func(test_prediction, target_reshaped)
        val_fn = theano.function([input_var, target_var], [test_prediction, val_loss])
    
    print "Starting data prefetcher"
    f = h5py.File(DATASET_TRAIN + ".hdf5")
    length = f["images"].shape[0]
    f.close()
    start_prefetching_thread(batch_size)
    if validate:
        print "Loading validation data"
    x_val, y_val = load_val_data()
    print "Starting training"
    train_losses = []
    val_losses = []
    val_errors = []
    bt = 0
    bidx = 0
    for epoch in range(num_epochs):
        if epoch == num_epochs // 2:
            print "Decreasing LR"
            sh_lr.set_value(lr / 10)
        # shuffle training data
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        print "Training Epoch %i" % (epoch + 1)

        # Train for one epoch
        for batch in iterate_minibatches_synchronized(length, length, batch_size, shuffle=True, augment=True, downsample=ds):
            inputs, targets = batch
            bts = time.time()
            err = train_fn(inputs, targets)
            bte = time.time()
            bt += (bte - bts)
            bidx += 1
            print inputs[0]
            print targets[0]
            if bidx == 60 and epoch == 0:
                tpb = bt / bidx
                print "Average time per forward/backward pass: " + str(tpb)
                eta = time.time() + num_epochs * (tpb * (length/batch_size))
                localtime = time.asctime( time.localtime(eta) )
                print "ETA: ", localtime

            train_losses.append(err)
            train_err += err
            train_batches += 1
            # Save intermedia train loss
            if bidx % 100 == 0:
                np.save('./data/' + name + '_epoch_' + str(num_epochs) + '_loss_train.npy', np.array(train_losses))



        # Validate model
        if validate:
            v_losses = []
            v_mse = []
            indices = np.arange(x_val.shape[0])
            for i in range(0,x_val.shape[0], batch_size):
                excerpt = indices[i:i+batch_size]
                x_in = x_val[excerpt,:,9:9+240, 12:12+320]
                y_in = y_val[excerpt,9:9+240:ds, 12:12+320:ds]
                v_pred, v_loss = val_fn(x_in, y_in)
                current_se = (np.exp(v_pred) - y_in) ** 2
                # calc current mse for this minibatch
                v_mse.append(current_se.mean())
                v_losses.append(v_loss)
            val_mse = np.array(v_mse).mean()
            val_loss = np.array(v_losses).mean()
            val_losses.append(val_loss)
            val_errors.append(val_mse)
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if validate:
            print("  val loss:\t\t{:.6f}".format(val_loss))
            print("  val mse:\t\t{:.6f}".format(val_mse))
        # Store intermediate val loss
        np.save('./data/' + name + '_epoch_' + str(num_epochs) + '_loss_val.npy', np.array(val_losses))
        np.save('./data/' + name + '_epoch_' + str(num_epochs) + '_errors_val.npy', np.array(val_errors))
    print "Saving data"
    # Save
    np.savez('./data/' + name +'_epoch_%i.npz' % num_epochs, *lasagne.layers.get_all_param_values(network))
    np.save('./data/' + name + '_epoch_' + str(num_epochs) + '_loss_train.npy', np.array(train_losses))
    


if __name__ == '__main__':
    main()

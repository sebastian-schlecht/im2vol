import numpy as np
np.random.seed(42)
import lasagne
from threading import Thread
import Queue
import theano
import theano.tensor as T
import h5py
import time, math
from PIL import Image
from scipy.ndimage.interpolation import zoom,rotate


from networks import  residual_unet, debug_net
from losses import spatial_gradient, berhu, mse

# Our shuffled dataset. Important that we store it in contigous blocks, and not chunks to have const. speed on variable batchsizes
DATASET_TRAIN = "/home/sebastianschlecht/depth_data/nyu_v1_shuffled_cc"

# Validation dataset that we keep in memory
DATASET_VAL = "/home/sebastianschlecht/depth_data/nyu_depth_v2_resized"
#DATASET_TRAIN=DATASET_VAL
# Name used for saving losses and models
name = "resunet_mse"
# Init with model params
model = None#'data/resunet_final_spatial_grad_epoch_45.npz'
# Synced mode for prefetching. Should stay True, otherwise this script will break
synced_prefetching = True
# Threadsafe queue for prefetching
q = Queue.Queue(maxsize=20)
# Validate during training
validate = True
# Use local assertions
assert_local = True

def c_assert(condition):
    """
    Conditional assert
    """
    if assert_local:
        assert condition



def prefetch_proc(bs, augment):
    """
    Open handle to DB and prefetch data. Also do random online augmention
    :return: None - Loops infinitely
    """
    
    def zoom_(ii,dd):
        f = np.random.randint(1050,1500) / 1000.
        h = int(dd.shape[0] / f)
        w = int(dd.shape[1] / f)
        
        s_fh = float(dd.shape[0]) / float(h)
        s_fw = float(dd.shape[1]) / float(w)

        s_f = (s_fh + s_fw) / 2.
        
        cy = np.random.randint(dd.shape[0] - h+1 )
        cx = np.random.randint(dd.shape[1] - w+1 )

        ddc = dd[cy:cy+h, cx:cx+w]
        iic = ii[:,cy:cy+h,cx:cx+w]
        dd_s = zoom(ddc,(s_fh, s_fw),order=0, prefilter=False)
        dd_s /= s_f
        ii_s = zoom(iic,(1,s_fh,s_fw),order=0, prefilter=False)
        return ii_s.astype(np.float32), dd_s.astype(np.float32)
        
    def exposure_transformer(img, level):
        """
        Change the exposure in an RGB image
        :param img: np.array
        :param level: Number
        :return: Image
        
        """
        def truncate(v):
            return 0 if v < 0 else 255 if v > 255 else v
        factor = (259. * (level+255.)) / (255. * (259.-level))
        for x in np.nditer(img, op_flags=['readwrite']):
            x[...] = truncate(factor * (x-128) + 128)
        return img
    
    def zoom_rot(ii,dd):
        """ Rotate and zoom an image around a given angle"""
        a = np.random.randint(-10,10)
        ddr = rotate(dd,a, order=0, prefilter=False)
        iir = rotate(ii.transpose((1,2,0)),a, order=0, prefilter=False)
        
        f = np.random.randint(10000,15100) / 10000.
        
        h = int(dd.shape[0] / f)
        w = int(dd.shape[1] / f)
        
        s_fh = float(dd.shape[0]) / float(h)
        s_fw = float(dd.shape[1]) / float(w)

        s_f = (s_fh + s_fw) / 2.
        
        offset  = 0
        cy = np.random.randint(offset,dd.shape[0] - h - offset + 1)
        cx = np.random.randint(offset,dd.shape[1] - w - offset + 1)

        ddc = ddr[cy:cy+h, cx:cx+w]
        iic = iir[cy:cy+h,cx:cx+w,:]

        dd_s = zoom(ddc,(s_fh, s_fw),order=0, prefilter=False)
        dd_s /= s_f
        ii_s = iic.transpose((2,0,1))
        
        ii_s = zoom(ii_s,(1,s_fh,s_fw),order=0, prefilter=False)
        
        return ii_s.astype(np.float32), dd_s.astype(np.float32)
    
    # Start func
    f = h5py.File(DATASET_TRAIN + ".hdf5")
    length = f["images"].shape[0]
    idx = 0
    # Subtract mean already and normalize std
    m = np.load(DATASET_TRAIN + ".npy").astype(np.float32)
    while True:
        if idx + bs > length:
            idx = 0
        images = np.array(f["images"][idx:idx+bs]).astype(np.float32).copy()
        labels = np.array(f["depths"][idx:idx+bs]).astype(np.float32).copy()
        idx += bs
        for i in range(images.shape[0]):
            # Flip horizontally with probability 0.5
            if augment:
                # Intensity
                lvl = np.random.randint(100,128)
                #images[i] = exposure_transformer(images[i], lvl)
                # Zoomrot
                images[i], labels[i] = zoom_rot(images[i], labels[i])
                # Flips
                p = np.random.randint(2)
                if p > 0:
                    images[i] = images[i, :, :, ::-1]
                    labels[i] = labels[i, :, ::-1]
                # RGB we mult with a random value between 0.9 and 1.1
                r = np.random.randint(90,111) / 100.
                g = np.random.randint(90,111) / 100.
                b = np.random.randint(90,111) / 100.
                images[i, 0] = (images[i, 0] * r).clip(0,255)
                images[i, 1] = (images[i, 1] * g).clip(0,255)
                images[i, 2] = (images[i, 2] * b).clip(0,255)
        #images[images > 255] = 255
            
            
        for i in range(images.shape[0]):
            pass
            # Subtract mean at the end and normalize variance
            images[i] = (images[i] - m) / 71.571201304890508    
        assert images.shape[0] == bs
        assert labels.shape[0] == bs
        q.put((images, labels), block=True)
        


def start_prefetching_thread(args):
    thread = Thread(target=prefetch_proc, args=args)
    thread.daemon = True
    thread.start()
    return thread


def iterate_minibatches_synchronized(inputlen, batchsize, augment=False, downsample=1):
    """
    Iterate minibatches from a synced queue
    :param inputlen:
    :param targetlen:
    :param shuffle:
    :param augment:
    :param downsample:
    :return:
    """
    for start_idx in range(0, inputlen - batchsize + 1, batchsize):
        # Random crops
        h = 228
        w = 304

        while True:
            try:
                inputs, targets = q.get(block=True,timeout=0.05)
                break
            except Queue.Empty:
                pass
                #print "Prefetch Queue Empty"
        
        # Random crop
        input_cropped = np.zeros((inputs.shape[0], inputs.shape[1],h,w), dtype=np.float32)
        target_cropped = np.zeros((targets.shape[0],h,w), dtype=np.float32)
        c_assert(inputs.shape[0] == targets.shape[0])
        for idx in range(inputs.shape[0]):
            if augment:
                cy = np.random.randint(inputs.shape[2] - h)
                cx = np.random.randint(inputs.shape[3] - w)
            else:
                cy = (inputs.shape[2] - h) // 2
                cx = (inputs.shape[3] - w) // 2
            input_cropped[idx] = inputs[idx, :, cy:cy+h, cx:cx+w]
            target_cropped[idx] = targets[idx, cy:cy+h, cx:cx+w]
        
        target_cropped = target_cropped[:,::downsample,::downsample]
        yield input_cropped, target_cropped



def load_val_data():
    f = h5py.File(DATASET_VAL + ".hdf5")

    x_train = np.array(f["images"]).astype(np.float32)
    y_train = np.array(f["depths"]).astype(np.float32)

    # Subtract train mean already and normalize std
    m = np.load(DATASET_TRAIN + ".npy").astype(np.float32)
    for i in range(len(x_train)):
        x_train[i] = (x_train[i] - m) / 71.571201304890508
    f.close()
    return (x_train, y_train)


def main(num_epochs=42, batch_size=16):
    loss_func = mse
    print "Building network"
    input_var = T.tensor4('inputs')
    target_var = T.tensor3('targets')
    # Reshape to enable usage in loss function
    target_reshaped = target_var.dimshuffle((0, "x", 1, 2))
    
    network = residual_unet(input_var=input_var)
    # Downsample factor
    ds = 2
    
    prediction = lasagne.layers.get_output(network)
    # Spatial grad / biweight / scale invariant error
    loss = loss_func(prediction, target_reshaped)
    # Add some L2
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    cost = loss + l2_penalty


    params = lasagne.layers.get_all_params(network, trainable=True)
    sh_lr = theano.shared(lasagne.utils.floatX(0.001))
    updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=sh_lr, momentum=0.95)
    # Load model weights
    if model is not None:
        print "Loading model weights %s" % model
        with np.load(model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
        
    print "Compiling training model"
    # We want to have the main loss only, not l2 values for recording
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    if validate:
        print "Compiling validation model"
        val_prediction = lasagne.layers.get_output(network, deterministic=True)
        val_loss = loss_func(val_prediction, target_reshaped)
        val_fn = theano.function([input_var, target_var], [val_prediction, val_loss])
    
    print "Starting data prefetcher"
    f = h5py.File(DATASET_TRAIN + ".hdf5")

    length = f["images"].shape[0]
    f.close()
    start_prefetching_thread((batch_size, True))
    if validate:
        print "Loading validation data"
        x_val, y_val = load_val_data()
    print "Starting training"
    train_losses = []
    val_losses = []
    val_errors = []
    bt = 0
    bidx = 0
    
    learning_rate_schedule = {
     0:  0.01, 
     1:  0.1,   
     6:  0.01,
     18: 0.001,
     30: 0.0001   
    }
    for epoch in range(num_epochs):
        if epoch in learning_rate_schedule:
            val = learning_rate_schedule[epoch]
            print "Setting LR to " + str(val)
            sh_lr.set_value(lasagne.utils.floatX(val))
            

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        print "Training Epoch %i" % (epoch + 1)

        # Train for one epoch
        for batch in iterate_minibatches_synchronized(length, batch_size, augment=True, downsample=ds):
            inputs, targets = batch
            bts = time.time()
            
            c_assert(inputs.dtype == np.float32)
            c_assert(targets.dtype == np.float32)
            
            err = train_fn(inputs, targets)
            bte = time.time()
            bt += (bte - bts)
            bidx += 1
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
                x_in = x_val[excerpt,:,6:6+228, 8:8+304].copy()
                y_in = y_val[excerpt,6:6+228, 8:8+304].copy()
                y_in = y_in[:,::ds,::ds]
                v_pred, v_loss = val_fn(x_in, y_in)
                y_t = y_in[:,np.newaxis,:,:]
                
                c_assert(y_t.shape == v_pred.shape)
                
                if loss_func == spatial_gradient:
                    v_pred = np.exp(v_pred)
                current_se = (v_pred - y_t) ** 2
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
        # Store intermediate weights
        np.savez('./data/' + name +'_epoch_%i.npz' % num_epochs, *lasagne.layers.get_all_param_values(network))
    print "Saving data"
    # Save
    np.savez('./data/' + name +'_epoch_%i.npz' % num_epochs, *lasagne.layers.get_all_param_values(network))
    np.save('./data/' + name + '_epoch_' + str(num_epochs) + '_loss_train.npy', np.array(train_losses))


if __name__ == '__main__':
    main()

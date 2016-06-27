import numpy as np
import os, sys
from skimage import io, transform
import h5py

sys.path.append('..')

# Dataset root
DATASET_PATH = '/Users/sebastian/Downloads/food-101/'

# Target file
TRAIN_PREFIX = "/Users/sebastian/Desktop/food101"

# Target size for transform
TARGET_SIZE = 255

# Total number of labels to expand the label tensor in
NUM_LABELS = 101

# Load train images
train_img_list = DATASET_PATH + 'meta/train.txt'
lines = [line.rstrip('\n') for line in open(train_img_list)]


# Prepare dir hash
dirs = {}
y = 0
for dirname in os.listdir( DATASET_PATH + 'images'):
    if not dirname.startswith((".")):
        dirs[dirname] = y
        y += 1

# Read images each by each and resize them into a
DB_FILE = TRAIN_PREFIX + '.hdf5'
f = h5py.File(DB_FILE)
# Read images into memory
print "Starting to process images"
idx = 0
images = np.zeros((len(lines), 3, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
labels = np.zeros((len(lines),), dtype=np.uint8)
for line in lines:
    # Process and reshape images
    print "Processing image %i out of %i" % (idx, len(lines))
    image = np.array(io.imread(DATASET_PATH + 'images/' + line + ".jpg"))
    rs = transform.resize(image, (TARGET_SIZE, TARGET_SIZE))
    # Handle grayscale
    if len(rs.shape) == 2:
        images[idx,0,:,:] = rs * 255
        images[idx,1,:,:] = rs * 255
        images[idx,2,:,:] = rs * 255
    else:
        images[idx] = rs.transpose((2, 0, 1)) * 255
    # Set label tensor
    label_name = line.partition("/")[0]
    label_idx = dirs[label_name]
    labels[idx] = label_idx
    # Increment index
    idx += 1


print "Storing images"
dimg = f.create_dataset("images", images.shape, dtype=images.dtype)
dimg[...] = images
print "Storing labels"
dlab = f.create_dataset("labels", labels.shape, dtype=labels.dtype)
dlab[...] = labels
f.close()


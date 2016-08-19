import sys
sys.path.append('..')

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

ds = "/home/ga29mix/data/converted_v2/"
tmp = "/tmp/merge_temp_h5"
target = "/home/ga29mix/data/nyu_v2"
files = [f for f in listdir(ds) if isfile(join(ds, f)) and f.endswith(".mat")]

truncate = 2
h = 240
w = 320

try:
    os.remove(tmp)
except:
    pass

try:
    os.remove(target + ".hdf5")
except:
    pass

# Read the first one
print ds + f[0]
f_1 = h5py.File(ds + files[0])
s_i = f_1["images"].shape
d_i = f_1["depths"].shape

print s_i
print d_i

# Create a new dataset
fo = h5py.File(tmp)
o_i = fo.create_dataset("images", (s_i[0] - truncate, s_i[1], s_i[2], s_i[3]),maxshape=(None, 3, h, w),dtype=np.uint8, chunks=True)
o_d = fo.create_dataset("depths", (d_i[0] - truncate, d_i[1], d_i[2]),maxshape=(None, h, w), dtype=np.float32, chunks=True)


# Write file 1
o_i[:] = np.array(f_1["images"][0:-truncate])
o_d[:] = np.array(f_1["depths"][0:-truncate])

# Write rest
rowptr = s_i[0] - truncate
for i in range(1, len(files)):
    print "Processing file %i out of %i" % (i, len(files))
    f_i = h5py.File(ds + files[i])
    s_i = f_i["images"].shape
    d_i = f_i["depths"].shape
    
    # Resize
    new_i = (o_i.shape[0] + s_i[0] - truncate, s_i[1], s_i[2], s_i[3])
    new_d = (o_d.shape[0] + d_i[0] - truncate, d_i[1], d_i[2])
    o_i.resize(new_i)
    o_d.resize(new_d)
    
    # Write
    o_i[rowptr:] = np.array(f_i["images"][0:-truncate])
    o_d[rowptr:] = np.array(f_i["depths"][0:-truncate])
    
    # Inc rowptr
    rowptr += (s_i[0] - truncate)
    f_i.close()
    
llen = fo["images"].shape[0]
fo.close()

# Create a move the content and thereby shuffle the data globally. images first to save RAM
print "Shuffling..."
perm = np.random.permutation(llen)

final = h5py.File(target + ".hdf5")
tempfile = h5py.File(tmp)


images = np.array(tempfile["images"])
images = images[perm]
final.create_dataset("images", data=images)

# Close and save RAM
final.close()
tempfile.close()

final = h5py.File(target + ".hdf5")
tempfile = h5py.File(tmp)

depths = np.array(tempfile["depths"])
depths = depths[perm]
final.create_dataset("depths", data=depths)
final.close()
tempfile.close()

try:
    os.remove(tmp)
except:
    pass


# Compute image mean
print "Computing image mean"
final = h5py.File(target + ".hdf5")
images = np.array(final["images"])
mean = images.mean(axis=0)
np.save(target + ".npy", mean)







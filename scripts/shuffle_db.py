import h5py as h
import numpy as np

IN_FILE = '/home/sebastianschlecht/depth_data/nyu_v1_large_cc.hdf5'
OUT_FILE = '/home/sebastianschlecht/depth_data/nyu_v1_shuffled_cc.hdf5'


f = h.File(IN_FILE)
llen = f["images"].shape[0]
perm = np.random.permutation(llen)

ii = np.array(f["images"])
iis = ii[perm]
fn = h.File(OUT_FILE)
i_s = fn.create_dataset("images",data=iis)
fn.close()
del ii
del iis
dd = np.array(f["depths"])
dds = dd[perm]
fn = h.File(OUT_FILE)
d_s = fn.create_dataset("depths", data=dds)
del dds
del dd
fn.close()
f.close()
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

nhits = hp.read_map("norm_nHits_SA_35FOV_ns512.fits")
nhits = hp.ud_grade(nhits,256)
mask_binary = (nhits > 0).astype(float)
hp.write_map("mask_binary.fits", mask_binary, dtype=np.single)
#hp.mollview(nhits)
#hp.mollview(mask_binary)
#plt.show()

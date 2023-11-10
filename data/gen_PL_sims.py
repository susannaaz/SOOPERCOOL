import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os

# Added SA
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--predir', dest='predir', 
                  default='home/susannaz/BBMASTER/data/',
                  type=str, help='Output directory')
(o, args) = parser.parse_args()
predir = o.predir

os.system(f'mkdir -p {predir}PL')
os.system(f'mkdir -p {predir}val')
os.system(f'mkdir -p {predir}CMBl')
os.system(f'mkdir -p {predir}CMBr')

nside = 64
ls = np.arange(3*nside)
cl = 1/(ls+10)**2
np.savez(f'{predir}PL/cl_PL.npz', ls=ls,
         clEE=cl, clEB=cl, clBE=cl, clBB=cl)

clEE = 1/(ls+10)**0.5
clBB = 0.05/(ls+10)**0.5
cl0 = np.zeros(3*nside)
np.savez(f'{predir}val/cl_val.npz', ls=ls,
         clEE=clEE, clEB=cl0, clBE=cl0, clBB=clBB)

d = np.loadtxt(f'{predir}camb_lens_nobb.dat', unpack=True)
clEE_CMBl = np.zeros(3*nside)
clEE_CMBl[1:] = (2*np.pi*d[2]/(d[0]*(d[0]+1)))[:3*nside-1]
clBB_CMBl = np.zeros(3*nside)
clBB_CMBl[1:] = (2*np.pi*d[3]/(d[0]*(d[0]+1)))[:3*nside-1]
np.savez(f'{predir}CMBl/cl_CMBl.npz', ls=ls,
         clEE=clEE_CMBl, clEB=cl0, clBE=cl0, clBB=clBB_CMBl)

dlr = np.loadtxt(f'{predir}camb_lens_r1.dat', unpack=True)
rfid = 0.01
clEE_CMBr = clEE_CMBl
clBB_CMBr = np.zeros(3*nside)
clBB_CMBr[1:] = (2*np.pi*rfid*(dlr[3]-d[3])/(d[0]*(d[0]+1)))[:3*nside-1]
np.savez(f'{predir}CMBr/cl_CMBr.npz', ls=ls,
         clEE=clEE_CMBr, clEB=cl0, clBE=cl0, clBB=clBB_CMBr)

nsims = 200
for i in range(nsims):
    print(i)

    # Fiducial PL sims
    seed = 1000+i
    alm = hp.synalm(cl)
    alm0 = alm*0
    mpE_Q, mpE_U = hp.alm2map_spin([alm, alm0], nside, spin=2, lmax=3*nside-1)
    mpB_Q, mpB_U = hp.alm2map_spin([alm0, alm], nside, spin=2, lmax=3*nside-1)
    hp.write_map(f'{predir}PL/plsim_{seed}_E.fits', [mpE_Q, mpE_U], overwrite=True,
                 dtype='float64')
    hp.write_map(f'{predir}PL/plsim_{seed}_B.fits', [mpB_Q, mpB_U], overwrite=True,
                dtype='float64')

    # Alternative PL sims
    almE = hp.synalm(clEE)
    almB = hp.synalm(clBB)
    mp_Q, mp_U = hp.alm2map_spin([almE, almB], nside, spin=2, lmax=3*nside-1)
    hp.write_map(f'{predir}val/valsim_{seed}.fits', [mp_Q, mp_U], overwrite=True,
                 dtype='float64')

    # CMB sims (lensing only with Al=1)
    almE = hp.synalm(clEE_CMBl)
    almB = hp.synalm(clBB_CMBl)
    mp_Q, mp_U = hp.alm2map_spin([almE, almB], nside, spin=2, lmax=3*nside-1)
    hp.write_map(f'{predir}CMBl/CMBl_{seed}.fits', [mp_Q, mp_U], overwrite=True,
                dtype='float64')

    # CMB sims (no lensing, r=0.01)
    almE = hp.synalm(clEE_CMBr)
    almB = hp.synalm(clBB_CMBr)
    mp_Q, mp_U = hp.alm2map_spin([almE, almB], nside, spin=2, lmax=3*nside-1)
    hp.write_map(f'{predir}CMBr/CMBr_{seed}.fits', [mp_Q, mp_U], overwrite=True,
                dtype='float64')



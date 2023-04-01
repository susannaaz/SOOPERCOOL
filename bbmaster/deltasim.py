import numpy as np
import healpy as hp
import pymaster as nmt
import copy
import sys


class DeltaBbl(object):
    def __init__(self, nside, dsim, filt, bins, lmin=2, lmax=None, pol=False, 
                 nsim_per_ell=10, seed0=1000, n_iter=0):
        if not isinstance(dsim, dict):
            raise TypeError("For now delta simulators can only be "
                            "specified through a dictionary.")

        if not isinstance(filt, dict):
            raise TypeError("For now filtering operations can only be "
                            "specified through a dictionary.")

        if not isinstance(bins, nmt.NmtBin):
            raise TypeError("`bins` must be a NaMaster NmtBin object.")
        self.dsim_d = copy.deepcopy(dsim)
        self.dsim = self._dsim_default
        self.filt_d = copy.deepcopy(filt)
        self.filt = self._filt_default
        self.lmin = lmin
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        if lmax is None:
            lmax = 3*self.nside-1
        self.lmax = lmax
        self.bins = bins
        self.n_bins = self.bins.get_n_bands()
        self.n_ells = lmax-lmin+1
        self.nsim_per_ell = nsim_per_ell
        self.seed0 = seed0
        self.n_iter = n_iter
        self.pol = pol
        self._prepare_filtering()
        self.alm_ord = hp.Alm()
        self._sqrt2 = np.sqrt(2.)
        self._oosqrt2 = 1/self._sqrt2
        
    def _prepare_filtering(self):
        # Match pixel resolution
        self.filt_d['mask'] = hp.ud_grade(self.filt_d['mask'], 
                                          nside_out=self.nside)

    def _gen_gaussian_map_old(self, ell):
        if self.pol:
            # We want to excite E (B) modes in Gaussian maps using healpy,
            # so we have to excite power spectra in EE and TE (BB and TB).
            # map_out contains 2x2 maps with (EE,EB,BE,BB)_in, (Q,U)_out
            map_out = np.zeros((4,2,self.npix))
            for ipol_in, p in enumerate([[1,3],[1,2,3,4,5],[1,2,3,4,5],[2,5]]):
                cl = np.zeros((6,3*self.nside), dtype='float64')
                for q in p:
                    cl[q, ell] = 1
                # synfast: Input Cls TT,EE,BB,TE,EB,TB. Output maps T,Q,U.
                map_out[ipol_in] = hp.synfast(cl, self.nside, pol=True, 
                                              new=False)[1:,:]
        else:
            cl = np.zeros(3*self.nside)
            cl[ell] = 1
            # TODO: we can save time on the SHT massively, since in this case 
            # there is no sum over ell!
            map_out = hp.synfast(cl, self.nside)
        return map_out
    
    def _rands_iterator(self):
        # Loops over polarization pair (ip), map (im) and yields two sets random
        # numbers to pick from (pk) for that specific combination. Example:
        # For EE (ip==0), pick same numbers (0) for E modes and different ones
        # for B (1,2)
        for ip, p in enumerate([[[0,1],[0,2]], [[0,1],[2,0]],
                                 [[0,1],[1,2]], [[0,1],[2,1]]]):
            for im, pk in enumerate(p):
                yield im, ip, pk
    
    def _gen_gaussian_map(self, ell):
        idx = self.alm_ord.getidx(3*self.nside-1, ell, np.arange(ell+1))
        if self.pol:
            # shape (map1,map2; EE,EB,BE,BB; Q,U; ipix)
            map_out = np.zeros((2,4,2,self.npix)) 
            alms = np.zeros((2,4,3,self.alm_ord.getsize(3*self.nside-1)),
                            dtype='complex128')
            # We only need to pick from three independent sets of alms
            rans = np.random.normal(0, self._oosqrt2,
                                    size=6*(ell+1)).reshape([3,2,ell+1])
            rans[:, 0, 0] *= self._sqrt2
            rans[:, 1, 0] = 0
            
            for im, ip, pk in self._rands_iterator():
                alms[im,ip,1,idx] = rans[pk[0],0] + 1j*rans[pk[0],1]
                alms[im,ip,2,idx] = rans[pk[1],0] + 1j*rans[pk[1],1]
                map_out[im,ip] = hp.alm2map(alms[im,ip], self.nside)[1:]
        else:
            alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),
                            dtype='complex128')
            rans = np.random.normal(0, self._oosqrt2, 
                                    size=2*(ell+1)).reshape([2,ell+1])
            rans[0, 0] *= self._sqrt2
            rans[1, 0] = 0
            alms[idx] = rans[0] + 1j*rans[1]
            # TODO: we can save time on the SHT massively, since in this case 
            # there is no sum over ell!
            map_out = hp.alm2map(alms, self.nside)
        return map_out

    def _gen_Z2_map(self, ell):
        # Analogous to Gaussian
        idx = self.alm_ord.getidx(3*self.nside-1, ell, np.arange(ell+1))
        if self.pol:
            # shape (map1,map2; EE,EB,BE,BB; Q,U; ipix)
            map_out = np.zeros((2,4,2,self.npix)) 
            alms = np.zeros((2,4,3,self.alm_ord.getsize(3*self.nside-1)),
                            dtype='complex128')
            # We only need to pick from three independent sets of alms
            rans = self._oosqrt2*(
                2*np.random.binomial(1,0.5,size=6*(ell+1))-1
            ).reshape([3,2,ell+1])
            rans[:, 0, 0] *= self._sqrt2
            rans[:, 1, 0] = 0
            
            for im, ip, pk in self._rands_iterator():
                alms[im,ip,1,idx] = rans[pk[0],0] + 1j*rans[pk[0],1]
                alms[im,ip,2,idx] = rans[pk[1],0] + 1j*rans[pk[1],1]
                map_out[im,ip] = hp.alm2map(alms[im,ip], self.nside)[1:]
        else:
            rans = self._oosqrt2*(
                2*np.random.binomial(1,0.5,size=2*(ell+1))-1
            ).reshape([2,ell+1])
            # Correct m=0 (it should be real and have twice as much variance)
            rans[0, 0] *= self._sqrt2
            rans[1, 0] = 0
            # Populate alms and transform to map
            alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),
                            dtype='complex128')
            alms[idx] = rans[0] + 1j*rans[1]
            # TODO: we can save time on the SHT massively, since in this case 
            # there is no sum over ell!
            map_out = hp.alm2map(alms, self.nside)
        return map_out
    
    def _gen_Z2_alm(self, ell):
        idx = self.alm_ord.getidx(3*self.nside-1, ell, 
                                  np.arange(ell+1))
        if self.pol:    # Z2 alms (TEB)        
            for ipol_in in range(2):
                alms = np.zeros((3, self.alm_ord.getsize(3*self.nside-1)),
                                dtype='complex128')
                rans = self._oosqrt2*(2*np.random.binomial(1,0.5, 
                                                           size=2*(ell+1))-1).reshape([2,ell+1])
                rans[0, 0] *= self._sqrt2
                rans[1, 0] = 0
                alms[ipol_in+1,idx] = rans[0] + 1j*rans[1]
        else:
            rans = self._oosqrt2*(2*np.random.binomial(1,0.5,
                                                       size=2*(ell+1))-1).reshape([2,ell+1])
            rans[0, 0] *= self._sqrt2
            rans[1, 0] = 0
            alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),
                            dtype='complex128')
            alms[idx] = rans[0] + 1j*rans[1]
        return alms

    def _dsim_default(self, seed, ell):
        np.random.seed(seed)
        if self.dsim_d['stats'] == 'Gaussian':
            return self._gen_gaussian_map(ell)
        elif self.dsim_d['stats'] == 'Z2':
            return self._gen_Z2_map(ell)
        else:
            raise ValueError("Only Gaussian and Z2 sims implemented")

    def _filt_default(self, mp_true):
        if self.pol:
            assert(mp_true.shape==(2,4,2,self.npix))
            map_out = self.filt_d['mask'][None,None,None,:]*mp_true
        else:
            map_out = self.filt_d['mask']*mp_true
        return map_out

    def gen_deltasim(self, seed, ell):
        dsim_true = self.dsim(seed, ell)
        dsim_filt = self.filt(dsim_true)
        return dsim_filt 
    
    def gen_deltasim_bpw(self, seed, ell):
        dsim = self.gen_deltasim(seed, ell)
        if self.pol:
            assert(dsim.shape==(2,4,2,self.npix))
            # cb has shape (EE,EB,BE,BB)_out, bpw_out, (EE,EB,BE,BB)_in
            cb = np.zeros((4,self.n_bins,4))
            pols = [(0,0),(0,1),(1,0),(1,1)]
            for ipol_in, (ip1,ip2) in enumerate(pols):
                tqu1 = np.concatenate((np.zeros((1,self.npix)), dsim[0,ipol_in]))
                tqu2 = np.concatenate((np.zeros((1,self.npix)), dsim[1,ipol_in]))
                
                # anafast only outputs EB, not BE for 2 polarized input maps. We
                # can generalize this using map2alm + alm2cl instead.
                alm1 = hp.map2alm(tqu1)[1:] # alm_EB
                alm2 = hp.map2alm(tqu2)[1:] # alm_EB
                for ipol_out, (iq1,iq2) in enumerate(pols):
                    cb[ipol_out, :, ipol_in] = self.bins.bin_cell(
                        hp.alm2cl(alm1[iq1], alm2[iq2])
                    )
        else:
            cb = self.bins.bin_cell(hp.anafast(dsim, iter=self.n_iter))
        return cb

    def gen_Bbl_at_ell(self, ell):
        if self.pol:
            # Bbl has shape pol_out, bpw_out, pol_in
            Bbl = np.zeros((4,self.n_bins,4))
        else:
            Bbl = np.zeros(self.n_bins)
        for i in range(self.nsim_per_ell):
            sys.stdout.write(f'\rell={ell}/{self.lmax}: sim {i} of {self.nsim_per_ell}')
            sys.stdout.flush()
            seed = ell*self.nsim_per_ell + i
            cb = self.gen_deltasim_bpw(seed, ell)
            Bbl += cb
        Bbl /= self.nsim_per_ell
        return Bbl

    def get_ells(self):
        return np.arange(self.lmin, self.lmax+1)

    def gen_Bbl_all(self):
        if self.pol:
            # arr_out has shape pol_out, bpw_out, pol_in, ell_in
            arr_out = np.zeros((4,self.n_bins,4,self.n_ells))
            for il in range(self.n_ells):
                arr_out[:,:,:,il] = self.gen_Bbl_at_ell(self.lmin+il)
        else: 
            arr_out = np.array([self.gen_Bbl_at_ell(l)
                         for l in self.get_ells()]).T
        return arr_out
import numpy as np
import healpy as hp
import pymaster as nmt
import copy


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

    def _gen_gaussian_alm(self, ell):
        if self.pol:
            # We want to excite E (B) modes in Gaussian maps using healpy,
            # so we have to excite power spectra in EE and TE (BB and TB).
            # map_out contains 2x2 maps with (E, B)_in, (Q, U)_out
            map_out = np.zeros((2,2,self.npix))
            for ipol_in, p in enumerate([[1,3],[2,5]]):
                cl = np.zeros((6,3*self.nside), dtype='float64')
                for q in p:
                    cl[q, ell] = 1
                # synfast: Input Cls are TT, EE, BB, TE, EB, TB. Output will be 
                # TQU maps.
                map_out[ipol_in] = hp.synfast(cl, self.nside, pol=True, 
                                              new=False)[1:,:]
        else:
            cl = np.zeros(3*self.nside)
            cl[ell] = 1
            map_out = hp.synfast(cl, self.nside)
        # TODO: we can save time on the SHT massively, since in this case there 
        # is no sum over ell!
        return map_out

    def _gen_Z2_alm(self, ell):
        idx = self.alm_ord.getidx(3*self.nside-1, ell, 
                                  np.arange(ell+1)) # shape (ell+1)
        # Generate Z2 numbers (one per m)
        # TODO: Is it clear that it's best to excite all m's rather than one 
        # (or some) at a time?
        if self.pol:
            # map_out contains 2x2 maps with (E, B)_in, (Q, U)_out
            map_out = np.zeros((2,2,self.npix)) 
            
            # Excite E and B individually
            for ipol_in in range(2):
                alms = np.zeros((3, self.alm_ord.getsize(3*self.nside-1)),
                                dtype='complex128')
                rans = self._oosqrt2*(2*np.random.binomial(1, 
                                                           0.5, 
                                                           size=2*(ell+1))-1).reshape([2,ell+1])
                rans[0, 0] *= self._sqrt2
                rans[1, 0] = 0
                alms[ipol_in+1,idx] = rans[0] + 1j*rans[1]
            
                # alm2map: Input alms are TEB. Output will be TQU maps.
                map_out[ipol_in] = hp.alm2map(alms, self.nside, pol=True)[1:]
        else:
            rans = self._oosqrt2*(2*np.random.binomial(1, 0.5,
                                                       size=2*(ell+1))-1).reshape([2,
                                                                                   ell+1])
            # Correct m=0 (it should be real and have twice as much variance)
            rans[0, 0] *= self._sqrt2
            rans[1, 0] = 0
            # Populate alms and transform to map
            # TODO: we can save time on the SHT massively, since in this case 
            # there is no sum over ell!
            alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),
                            dtype='complex128')
            alms[idx] = rans[0] + 1j*rans[1]
            map_out = hp.alm2map(alms, self.nside)
        return map_out

    def _dsim_default(self, seed, ell):
        np.random.seed(seed)
        if self.dsim_d['stats'] == 'Gaussian':
            return self._gen_gaussian_alm(ell)
        elif self.dsim_d['stats'] == 'Z2':
            return self._gen_Z2_alm(ell)
        else:
            raise ValueError("Only Gaussian and Z2 sims implemented")

    def _filt_default(self, mp_true):
        if self.pol:
            assert(mp_true.shape==(2,2,self.npix))
            map_out = self.filt_d['mask'][None,None,:]*mp_true
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
            assert(dsim.shape==(2,2,self.npix))
            # cb contains 2x2 sets of bandpowers with (EE, BB)_in, 
            # (EE, BB)_out
            # TODO: Cross spectra between dsim1 and dsim2, giving EB and BE
            cb = np.zeros((2,2,self.n_bins))
            for ipol_in in range(2):
                # anafast: Inputs are TQU maps. Cls are TT, EE, BB, TE, EB, TB.
                dsim_tqu = np.concatenate((np.zeros((1,self.npix)), 
                                           dsim[ipol_in]))
                cl = hp.anafast(dsim_tqu, pol=True,
                                iter=self.n_iter)[[1,2]]
                for ipol_out in range(2):
                    cb[ipol_in, ipol_out] = self.bins.bin_cell(cl[ipol_out])
        else:
            cb = self.bins.bin_cell(hp.anafast(dsim, iter=self.n_iter))
        return cb

    def gen_Bbl_at_ell(self, ell):
        if self.pol:
            # Bbl contains 2x2 sets of bandpowers with (EE, BB)_in, 
            # (EE, BB)_out
            Bbl = np.zeros((2,2,self.n_bins))
        else:
            Bbl = np.zeros(self.n_bins)
        for i in range(self.nsim_per_ell):
            seed = ell*self.nsim_per_ell + i
            cb = self.gen_deltasim_bpw(seed, ell)
            Bbl += cb
        Bbl /= self.nsim_per_ell
        return Bbl

    def get_ells(self):
        return np.arange(self.lmin, self.lmax+1)

    def gen_Bbl_all(self):
        if self.pol:
            arr_out = np.zeros((2,self.n_ells,2,self.n_bins))
            for il in range(self.n_ells):
                arr_out[:,il,:,:] = self.gen_Bbl_at_ell(self.lmin+il)
        else: 
            arr_out = np.array([self.gen_Bbl_at_ell(l)
                         for l in self.get_ells()]).T
        return arr_out
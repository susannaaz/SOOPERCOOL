import numpy as np
import healpy as hp
import pymaster as nmt
import copy


class DeltaBbl(object):
    def __init__(self, nside, dsim, filt, bins, lmin=2, lmax=None, pol=True, nsim_per_ell=10, seed0=1000, n_iter=0):
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
        self.filt_d['mask'] = hp.ud_grade(self.filt_d['mask'], nside_out=self.nside)

    def _gen_gaussian_alm(self, ell):
        if self.pol:
            # If we want to excite X modes (X=E,B), then we have to excite TX too
            # TODO: Do we want to excite EB (meaning E AND B in maps) too?
            map_out = np.zeros((2,self.npix)) # E, B
            # Excite EE, BB in maps, given CL ordered as TT, EE, BB, TE, EB, TB.
            for ip, p in enumerate([1,3],[2,5]):  
                cl = np.zeros(6, 3*self.nside)
                for q in p:
                    cl[q, ell] = 1
                map_out[ip] = hp.synfast(cl, nside, pol=True, new=False)
        else:
            cl = np.zeros(3*self.nside)
            cl[ell] = 1
            map_out = hp.synfast(cl, self.nside)
        # TODO: we can save time on the SHT massively, since in this case there is no
        # sum over ell!
        # Q: Write this simpler SHT by ourselves, or use existing healpy functionality?
        return map_out

    def _gen_Z2_alm(self, ell):
        if self.pol:
            raise NotImplementedError('Polarized Z2 alms are yet to be implemented.')
        else:
            idx = self.alm_ord.getidx(3*self.nside-1, ell,
                                      np.arange(ell+1))
            # Generate Z2 numbers (one per m)
            # TODO: Is it clear that it's best to excite all m's
            # rather than one (or some) at a time?
            rans = self._oosqrt2*(2*np.random.binomial(1, 0.5,
                                                       size=2*(ell+1))-1).reshape([2,
                                                                                   ell+1])
            # Correct m=0 (it should be real and have twice as much variance
            rans[0, 0] *= self._sqrt2
            rans[1, 0] = 0
            # Populate alms and transform to map
            # TODO: we can save time on the SHT massively, since in this case there is no
            # sum over ell!
            alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),
                            dtype='complex128')
            alms[idx] = rans[0]+1j*rans[1]
            map_out = hp.alm2map(alms, self.nside)
        return map_out

    def _dsim_default(self, seed, ell):
        np.random.seed(seed)
        if self.dsim_d['stats'] == 'Gaussian':
            return self._gen_gaussian_alm(ell)
        elif self.dsim_d['stats'] == 'Z2':
            return self._gen_Z2_alm(ell)
        else:
            raise ValueError("Only Gaussian sims implemented")

    def _filt_default(self, mp_true):
        if self.pol:
            assert(mp_true.shape==(2,self.npix))
            map_out = [self.filt_d['mask']*mt for mt in mp_true]
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
            idx_pol =[1,2,4] # EE, BB, EB
            cb = self.bins.bin_cell(hp.anafast([np.zeros(self.npix)]+dsim, iter=self.n_iter)[idx_pol])
        else:
            cb = self.bins.bin_cell(hp.anafast(dsim, iter=self.n_iter))
        return cb

    def gen_Bbl_at_ell(self, ell):
        if self.pol:
            # TODO: Decouple seeds for EE, BB, EB ?
            Bbl = np.zeros((3,self.bins.get_n_bands()))
        else:
            Bbl = np.zeros(self.bins.get_n_bands())
        for i in range(self.nsim_per_ell):
            seed = ell*self.nsim_per_ell + i
            cb = self.gen_deltasim_bpw(seed, ell)
            Bbl += cb
        Bbl /= self.nsim_per_ell
        return Bbl

    def get_ells(self):
        return np.arange(self.lmin, self.lmax+1)

    def gen_Bbl_all(self):
        if self.pol: # shape (n_b, n_l, n_p)
            arr_out = np.array([self.gen_Bbl_at_ell(l)
                         for l in self.get_ells()]).reshape(self.bins.get_n_bands(), self.n_ells, -1)
        else: # (n_b, n_l)
            arr_out = np.array([self.gen_Bbl_at_ell(l)
                         for l in self.get_ells()]).T 
        return arr_out

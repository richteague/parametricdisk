"""
A standard protoplanetary disk model following the likes of Williams and Best
(2014). There are two ways to play with the flaring, either through the
temperature structure or the density structure.

Temperautre method: using the parameters 'Htmp0' and 'Htmpq' to describe the
transition between the atmospheric temperature and midplane temperature as a
power-law as done in Rosenfeld et al. (2013). If this are both not set, it will
revert to assuming this occurs at four pressure scale heights.

Density method: use the parameters 'Hdns0' and 'Hdnsq' to describe the scale
height of the disk as a power-law, akin to Trapmann et al. (2017). This
requires dfunc = 'gaussian' otherwise these parameters are ignored.
"""

import warnings
import numpy as np
import scipy.constants as sc
from limepy.analysis.collisionalrates import ratefile
warnings.filterwarnings('ignore')


class ppd:

    msun = 1.988e30     # Stellar mass.
    mu = 2.35           # Mean molecular weight.
    B0 = 5.7635968e10   # Rotational constant.
    rates_path = '/Users/rteague/PythonPackages/limepy/aux/'

    def __init__(self, **kwargs):
        """Protoplanetary disk model."""

        # Model grid - cartesian grid.
        # Can specify the inner and outer radii with the number of grid points.
        # By default the grid will sample 500 points between [0, 7] * rval0 in
        # the radial direction and [0, 5] * rval0 in the vertical.
        # TODO: Include the option to have logarithmic values.

        self.rval0 = kwargs.get('rval0', 35.)
        self.rmin = kwargs.get('rmin', 0.)
        self.rmax = kwargs.get('rmax', 6. * self.rval0)
        self.nrpnts = kwargs.get('nr', 500)
        self.zmin = kwargs.get('zmin', 0.)
        self.zmax = kwargs.get('zmax', 5. * self.rval0)
        self.nzpnts = kwargs.get('nz', 500)
        self.rvals = np.linspace(self.rmin, self.rmax, self.nrpnts)
        self.zvals = np.linspace(self.zmin, self.zmax, self.nzpnts)
        self.rvals_m = self.rvals * sc.au
        self.zvals_m = self.zvals * sc.au
        self.rvals_cm = self.rvals_m * 1e2
        self.zvals_cm = self.zvals_m * 1e2

        # System masses. In solar masses.

        self.mstar = kwargs.get('mstar', 1.0)

        # Surface density - assumes a self-similar solution. The
        # normalisation of which can be controlled via either the 'sigm0'
        # parameter or the 'mdisk' value. The former is in [g/cm^2] and the
        # latter in [Msun]. Gaps can be added in as a list of lists. The gaps
        # kwarg should be a list of gap descriptions:v[center, width, depth]
        # where the centre and width are in [au], the depth relative to the
        # unperturbed surface density. TODO: Check how many gaps are input.

        self.mdisk = kwargs.get('mdisk', None)
        self.sigm0 = kwargs.get('sigm0', 15.)
        self.sigmq = kwargs.get('sigmq', -1.)

        if self.mdisk is not None:
            q = (2. + self.sigmq)
            self.sigm0 = self.mdisk * q * self.msun * 1e3
            self.sigm0 /= 2. * np.pi * np.power(self.rval0 * sc.au * 1e2, 2)
            self.sigm0 /= np.exp(np.power(self.rvals[0] / self.rval0, q))
        else:
            q = (2. + self.sigmq)
            self.mdisk = self.sigm0 / q / self.msun / 1e3
            self.mdisk *= 2. * np.pi * np.power(self.rval0 * sc.au * 1e2, 2)
            self.mdisk *= np.exp(np.power(self.rvals[0] / self.rval0, q))

        self.gaps = kwargs.get('gaps', None)
        if self.gaps is not None:
            self.gap_profile = []
            for gap in self.gaps:
                x0, dx, ds = gap[:3]
                profile = self._gaussian(self.rvals, x0, dx, ds)
                self.gap_profile += [1. - profile]
            self.gap_profile = np.prod(self.gap_profile, axis=0)
        else:
            self.gap_profile = np.ones(self.nrpnts)
        if self.gap_profile.shape != self.rvals.shape:
            raise ValueError("Wrong gap profile shape.")

        self.sigm = self._calc_surfacedensity() * self.gap_profile
        self.mdisk = self._calc_mdisk()

        # Temperature - two layer approximation from Dartois et al. (2013).
        # The {Htmp0, Htmpq} values describe the transition between the
        # midplane and atmospheric temperature regime. By default, this will
        # use Zq = 4Hp, where Hp is the pressure scale height.

        self.tmid0 = kwargs.get('tmid0', 20.)
        self.tatm0 = kwargs.get('tatm0', 90.)
        self.tmidq = kwargs.get('tmidq', -0.3)
        self.tatmq = kwargs.get('tatmq', -0.3)
        self.Htmp0 = kwargs.get('Htmp0', None)
        self.Htmpq = kwargs.get('Htmpq', None)

        # Volume density - by default the vertical structure is hydrostatic.
        # The {Hdns0, Hdnsq} parameters descibe the density scale height. If
        # dfunc='gaussian' then this will set the scaleheight used for that.
        # If none are specified, the pressure scale height is used by deafult.
        # A minimum density can be set, useful for trimming large models.

        self.Hdns0 = kwargs.get('Hdns0', None)
        self.Hdnsq = kwargs.get('Hdnsq', None)
        self.minH2 = kwargs.get('minH2', 0.0)

        flag = False
        if self.Htmp0 is not None and self.Htmpq is not None:
            flag = True
        if self.Hdns0 is not None and self.Hdnsq is not None:
            if flag:
                raise ValueError("All four scale height parameters set.")

        self.dfunc = kwargs.get('dfunc', 'hydrostatic').lower()
        if self.dfunc not in ['gaussian', 'hydrostatic']:
            raise ValueError("dfunc must be 'gaussian' or 'hydrostatic'.")
        else:
            if self.dfunc == 'gaussian':
                self._calc_volumedensity = self._calc_volumedensity_gaussian
            else:
                self._calc_volumedensity = self._calc_volumedensity_hydrostatic

        self.tmid = self._calc_midplanetemp()
        self.tatm = self._calc_atmospheretemp()
        self.temp = self._calc_temperature()

        self.dens = self._calc_volumedensity()
        self.Hp = self._calc_scaleheight()

        # CO distribution. Assume that xgas is a radial function.

        self.xgas = self._make_profile(kwargs.get('xgas', 1e-4))
        self.xdep = self._make_profile(kwargs.get('xdep', 1e-8))
        self.ndiss = kwargs.get('ndiss', 1.3e21)
        self.tfreeze = kwargs.get('tfreeze', 19.)

        self.abun = self._calc_abundance()
        self.column = self._calc_columndensity()

        # Radiative transfer.

        self.molecule = kwargs.get('molecule', 'CO').lower()
        self.rates = ratefile('%s%s.dat' % (self.rates_path, self.molecule))
        self.vturb = kwargs.get('vturb', 0.0)

        # Rotation velocity - by default just Keplerian rotation including the
        # height above the midplane. Optionally can include the disk mass in
        # the central mass calculation, or the pressure support.

        self.incl_altitude = kwargs.get('incl_altitude', True)
        self.incl_diskmass = kwargs.get('incl_diskmass', False)
        self.incl_pressure = kwargs.get('incl_pressure', True)

        self.rotation = self._calc_rotation()

        # Others.

        self.verbose = kwargs.get('verbose', False)
        return

    def _calc_rotation(self):
        """Calculate the azimuthal rotation velocity [m/s]."""

        # Keplerian rotation.
        if self.incl_altitude:
            vkep = np.hypot(self.rvals_m[None, :], self.zvals_m[:, None])
        else:
            vkep = self.rvals_m[None, :] * np.ones(self.nzpnts)[:, None]
        vkep = np.power(self.rvals_m, 2)[None, :] * np.power(vkep, -3)
        vkep *= sc.G * self.mstar * self.msun

        # Disk gravity component.
        if self.incl_diskmass:
            raise NotImplementedError()
        else:
            vgrav = 0.0

        # Pressure gradient.
        if self.incl_pressure:
            n = self.dens * 1e6
            P = n * sc.k * self.temp
            dPdr = np.gradient(P, self.rvals_m, axis=1)
            vpres = self.rvals_m[None, :] * dPdr / n / sc.m_p / self.mu
        else:
            vpres = 0.0

        # Combined.
        return np.sqrt(vkep + vgrav + vpres)

    def _gaussian(self, x, x0, dx, A=None):
        """Gaussian function. If A is None, normalize it. dx is stdev."""
        if A is None:
            A = 1. / np.sqrt(2. * np.pi) / dx
        return A * np.exp(-0.5 * np.power((x - x0) / dx, 2))

    def _calc_mdisk(self):
        """Return Mdisk in [Msun]."""
        sigm = np.where(np.isfinite(self.sigm), self.sigm, 0.0)
        mdisk = np.trapz(sigm * self.rvals_cm, self.rvals_cm)
        return 2. * np.pi * mdisk / 1e3 / self.msun

    def radialpowerlaw(self, x0, q):
        """Radial power law."""
        return x0 * np.power(self.rvals / self.rval0, q)

    def _calc_surfacedensity(self):
        """Surface density in [g / cm^2]."""
        sigm = self.radialpowerlaw(self.sigm0, self.sigmq)
        sigm *= np.exp(self.radialpowerlaw(-1.0, 2.+self.sigmq))
        return np.where(self.rvals >= self.rmin, sigm, 0.0)

    def _calc_columndensity(self):
        """Column density of CO in [/cm^2]."""
        return np.trapz(self.abun*self.dens, x=self.zvals*sc.au*1e2, axis=0)

    def _calc_midplanetemp(self):
        """Midplane temperature in [K]."""
        return self.radialpowerlaw(self.tmid0, self.tmidq)

    def _calc_atmospheretemp(self):
        """Atmospheric temperature in [K]."""
        return self.radialpowerlaw(self.tatm0, self.tatmq)

    def _calc_scaleheight(self, tmid=None):
        """Scale height in [au]."""
        if self.dfunc == 'gaussian':
            if self.Hdns0 is not None and self.Hdnsq is not None:
                return self.radialpowerlaw(self.Hdns0, self.Hdnsq)
        elif (self.Hdns0 is None) != (self.Hdnsq is None):
            if self.verbose:
                print("Both 'Hdns0' and 'Hdnsq' must be set.")
                print("Reverting to pressure scale height.")
        if tmid is None:
            tmid = self._calc_midplanetemp()
        Hp = sc.k * tmid * np.power(self.rvals, 3) / self.mu / sc.m_p
        return np.sqrt(Hp * sc.au / sc.G / self.mstar / self.msun)

    def _calc_temperature(self, tmid=None, tatm=None):
        """Gas temperature in [K]."""
        if tmid is None:
            tmid = self.tmid
        if tatm is None:
            tatm = self.tatm
        if (self.Htmp0 is None) != (self.Htmpq is None):
            if self.verbose:
                print("Both 'Htmp0' and 'Htmpq' must be set.")
                print("Reverting to Zq is four pressure scale heights.")
        if self.Htmp0 is not None and self.Htmpq is not None:
            zq = self.radialpowerlaw(self.Htmp0, self.Htmpq)
        else:
            zq = 4 * self._calc_scaleheight(tmid=tmid)
        T = np.cos(np.pi * self.zvals[:, None] / 2. / zq[None, :])**2
        T = T * (tmid - tatm)[None, :] + tatm[None, :]
        return np.where(self.zvals[:, None] >= zq[None, :], tatm[None, :], T)

    def _calc_soundspeed(self, temp=None):
        """Sound speed in [m/s]."""
        if temp is None:
            temp = self._calc_temperature()
        return np.sqrt(sc.k * temp / self.mu / sc.m_p)

    def _calc_volumedensity_normalise(self, rho, sigma=None):
        """Normalised the density profile."""
        if sigma is None:
            sigma = self.sigm
        rho *= sigma[None, :] / np.trapz(rho, self.zvals_cm, axis=0)
        rho /= self.mu * sc.m_p * 2e3
        mask = np.logical_and(np.isfinite(rho), rho >= self.minH2)
        return np.where(mask, rho, 0.0)

    def _calc_volumedensity_hydrostatic(self, temp=None, sigma=None):
        """Hydrostatic density distribution [/cm^3]."""
        if temp is None:
            temp = self.temp
        if sigma is None:
            sigma = self.sigm
        dT = np.diff(np.log(temp), axis=0)
        dT = np.vstack((dT, dT[-1]))
        dz = np.diff(self.zvals * sc.au)
        dz = np.hstack((dz, dz[-1]))
        cs = dz[:, None] * np.power(self._calc_soundspeed(temp), -2.)[None, :]
        G = (np.hypot(self.rvals[None, :], self.zvals[:, None]) * sc.au)**3
        G = sc.G * self.mstar * self.msun * self.zvals[:, None] * sc.au / G
        drho = np.squeeze(dT + cs * G)
        rho = np.ones(temp.shape)
        for i in range(self.rvals.size):
            for j in range(1, self.zvals.size):
                rho[j, i] = np.exp(np.log(rho[j-1, i]) - drho[j, i])
        return self._calc_volumedensity_normalise(rho, sigma)

    def _calc_volumedensity_gaussian(self, temp=None, sigma=None):
        """Gaussian density distribution [/cm^3]."""
        if temp is None:
            temp = self.temp
        if sigma is None:
            sigma = self.sigm
        Hp = self._calc_scaleheight(temp[abs(self.zvals).argmin()])
        rho = np.exp(-0.5 * (self.zvals[:, None] / Hp[None, :])**2)
        rho /= np.sqrt(2. * np.pi) * Hp * sc.au * 100.
        return self._calc_volumedensity_normalise(rho, sigma)

    def _calc_abundance(self, temp=None, dens=None):
        """Returns molecular abundance [wrt H2]."""
        if temp is None:
            temp = self.temp
        if dens is None:
            dens = self.dens
        dz = abs(np.diff(self.zvals).mean()) * sc.au * 1e2
        col = np.cumsum(dens[::-1], axis=0)[::-1] * dz
        mask = np.logical_and(col > self.ndiss, temp > self.tfreeze)
        return np.where(mask, self.xgas[None, :], self.xdep[None, :])

    def _sample_input(self, arr_y):
        """Sample 'arr' across the model radius."""
        arr_x = np.linspace(self.rmin, self.rmax, len(arr_y))
        return np.interp(self.rvals, arr_x, arr_y)

    def _make_profile(self, value):
        """Make the input into a radial profile."""
        if type(value) == float:
            return np.array([value for _ in range(self.nrpnts)])
        return self._sample_input(value)

    # Functions to do simple radiative transfer.

    def _calc_Nu(self, J=0):
        """Return the column density in each cell of that level [/cm^2]."""
        Eu = self.rates.lines[J+1].Eup
        gu = self.rates.levels[J+2].g
        abun = gu * self.abun * self.dens / self._calc_Qrot(self.temp)
        dz = np.diff(self.zvals_cm)
        dz = np.insert(dz, 0, dz[0])
        return abun * np.exp(-Eu / self.temp) * dz[:, None]

    def _calc_dV(self):
        """Return the Doppler width of the line at each cell."""
        vtherm = np.sqrt(2. * sc.k * self.temp / self.rates.mu / sc.m_p)
        return np.hypot(vtherm, self.vturb)

    def _calc_FWHM(self):
        """Return the FWHM of the line at each cell."""
        return 2. * np.sqrt(np.log(2.)) * self._calc_dV()

    def _calc_tau(self, J=0):
        """Return the optical depth of each cell for the given transition."""
        Nu = self._calc_Nu(J=J) * 1e4
        FWHM = self._calc_FWHM()
        Au = self.rates.lines[J+1].A
        nu = self.rates.lines[J+1].freq
        tau = Nu * Au * sc.c**3 / 8. / np.pi / nu**3 / FWHM
        return tau * (np.exp(sc.h * nu / sc.k / self.temp) - 1.)

    def _calc_ctau(self, J=0):
        """Return (1 - exp(-tau)) for each cell."""
        tau = self._calc_tau(J=J)
        return np.cumsum(tau[::-1], axis=0)[::-1]

    def _calc_Tmb(self, J=0):
        """Return the observed main beam temperature [K]."""
        return self.temp * (1. - np.exp(-self._calc_ctau(J=J)))

    def _calc_Qrot(self, T):
        """Rotational parition function."""
        return sc.k * T / sc.h / self.B0 + 1. / 3.

    def _get_tau_indices(self, tau=1.0, J=0):
        """Return the indices of the tau surface."""
        ctau = self._calc_ctau(J=J)
        idx = np.argmin(abs(ctau - np.ones(self.nrpnts) * tau), axis=0)
        return np.where(ctau[0] > tau, idx, 0)

    def tau_emission(self, tau=1.0, J=0):
        """Return emission profile at the surface where tau is reached [K]."""
        Tmb = self._calc_Tmb(J=J)
        idxs = self._get_tau_indices(tau=tau, J=J)
        return np.squeeze([Tmb[idx, i] for i, idx in enumerate(idxs)])

    def tau_temperature(self, tau=1.0, J=0):
        """Return the temperature at the tau surface [K]."""
        idxs = self._get_tau_indices(tau=tau, J=J)
        return np.squeeze([self.temp[idx, i] for i, idx in enumerate(idxs)])

    def tau_surface(self, tau=1.0, J=0):
        """Return the height of the tau surface [au]."""
        idxs = self._get_tau_indices(tau=tau, J=J)
        return np.squeeze([self.zvals[idx] for idx in idxs])

    def emission(self, J=0):
        """Return the radial emission profile [K]."""
        Tmb = self._calc_Tmb(J=J)
        weights = (1. - np.exp(-self._calc_ctau(J=J)))
        weights += 1e-50 * np.random.randn(weights.size).reshape(weights.shape)
        return np.average(Tmb, weights=weights, axis=0)

    # Functions to write the model out to limepy or similar.

    def write_header(self, filename, mincolumn=None):
        """Write the model to a .h file for LIME."""

        # Flatten all the arrays to save.
        rpnts = self.rvals[None, :] * np.ones(self.nzpnts)[:, None]
        zpnts = np.ones(self.nrpnts)[None, :] * self.zvals[:, None]
        rpnts = rpnts.T.flatten()
        zpnts = zpnts.T.flatten()
        dflat = self.dens.T.flatten()
        tflat = self.temp.T.flatten()
        aflat = self.abun.T.flatten()

        # If requested, remove radial points where the CO column density is
        # too low specified by the 'mincolumn' value.
        if mincolumn is not None:
            cflat = self.column[None, :] * np.ones(self.nzpnts)[:, None]
            disk = cflat.T.flatten() >= mincolumn
            rpnts = rpnts[disk]
            zpnts = zpnts[disk]
            dflat = dflat[disk]
            tflat = tflat[disk]
            aflat = aflat[disk]

        # Remove any points outside the disk.
        disk = dflat >= self.minH2
        rpnts = rpnts[disk]
        zpnts = zpnts[disk]
        dflat = dflat[disk]
        tflat = tflat[disk]
        aflat = aflat[disk]

        # Remove any regions with zero or NaN temperatures.
        disk = np.logical_and(np.isfinite(tflat), tflat > 0.0)
        rpnts = rpnts[disk]
        zpnts = zpnts[disk]
        dflat = dflat[disk]
        tflat = tflat[disk]
        aflat = aflat[disk]

        # Remove all points where z or r are negative.
        disk = np.logical_and(rpnts > 0, zpnts >= 0)
        rpnts = rpnts[disk]
        zpnts = zpnts[disk]
        dflat = dflat[disk]
        tflat = tflat[disk]
        aflat = aflat[disk]

        # Change to LIME appropriate units.
        dflat *= 1e6

        # Write the strings to a single file.
        arrays = np.nan_to_num([rpnts, zpnts, dflat, tflat, aflat])
        arraynames = ['c1arr', 'c2arr', 'dens', 'temp', 'abund']
        hstring = ''
        for array, name in zip(arrays, arraynames):
            hstring += self._write_header_string(array, name)
        with open('%s.h' % filename.replace('.h', ''), 'w') as hfile:
            hfile.write('%s' % hstring)
        print 'Written to %s.h.' % filename.replace('.h', '')
        return

    def _write_header_string(self, array, name):
        """Returns a string of the array to save."""
        tosave = 'const static double %s[%d] = {' % (name, array.size)
        for val in array:
            tosave += '%.3e, ' % val
        tosave = tosave[:-2] + '};\n'
        return tosave

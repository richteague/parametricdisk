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
from scipy.interpolate import griddata
warnings.filterwarnings('ignore')


class ppd:

    msun = 1.988e30     # Stellar mass.
    mu = 2.35           # Mean molecular weight.
    B0 = 5.7635968e10   # CO rotational constant.
    rates_path = '/Users/rteague/PythonPackages/limepy/aux/'

    def __init__(self, **kwargs):
        """Protoplanetary disk model."""
        self.verbose = kwargs.get('verbose', True)

        # Model grid - cartesian grid.
        # Can specify the inner and outer radii with the number of grid points.
        # By default the grid will sample 500 points between [0, 7] * rval0 in
        # the radial direction and [0, 5] * rval0 in the vertical.
        # TODO: Include the option to have logarithmic values.

        self.rval0 = kwargs.get('rval0', 20.)
        self.rmin = kwargs.get('rmin', 0.)
        self.rmax = kwargs.get('rmax', 6. * self.rval0)
        self.nr = kwargs.get('nr', 500)

        self.zmin = kwargs.get('zmin', 0.)
        self.zmax = kwargs.get('zmax', 3. * self.rval0)
        self.nz = kwargs.get('nz', 500)

        self.rvals = np.linspace(self.rmin, self.rmax, self.nr)
        self.zvals = np.linspace(self.zmin, self.zmax, self.nz)
        self.rpnts = self.rvals[None, :] * np.ones(self.nz)[:, None]
        self.zpnts = np.ones(self.nr)[None, :] * self.zvals[:, None]

        self.rvals_m = self.rvals * sc.au
        self.zvals_m = self.zvals * sc.au
        self.rvals_cm = self.rvals_m * 1e2
        self.zvals_cm = self.zvals_m * 1e2

        # System masses. In solar masses.

        self.mstar = kwargs.get('mstar', 1.0)

        # Surface density of total gas - [g/cm^2]. Different ways to specify:
        #
        #   1 - Self-similar solution (Lynden-Bell & Pringle 1974). This is the
        #       default mode. The normalization can be given either as third of
        #       the surface density at `rval0` through `sigm0`, or with a disk
        #       mass through `mdisk`. Note: if `mdisk` is given, this will
        #       overwrite the `sigm0` value.
        #
        #   2 - Presscribed surface density. This allows models to be easily
        #       modelled. The input needs to be an array so that it can be
        #       interpolated between. It is assumed that the array linearlly
        #       samples the radius between `rmin` and `rmax`. If this is used
        #       then `sigm0` and `sigmq` have no meaning, while `mdisk` is
        #       calculated afterwards.
        #
        # Gaps in the disk can also be specified with `gaps`. This should be a
        # list of gap descriptions: [center, width, depth] where the centre and
        # width are in [au] and the depth relative to the unperturbed surface
        # density at the gap center. If gaps are specified, `mdisk` will be
        # recalculated.
        #
        # TODO: allow the disk mass to remain constant after gaps are added.

        self.sigm0 = kwargs.get('sigm0', 15.)
        self.mdisk = kwargs.get('mdisk', None)
        self.sigma = kwargs.get('sigma', None)
        self.sigmq = kwargs.get('sigmq', kwargs.get('gamma', 1.))

        if self.sigma is not None:
            self.sigma = self._make_profile(self.sigma)
        elif self.mdisk is not None:
            q = (2. - self.sigmq)
            self.sigm0 = self.mdisk * q * self.msun * 1e3
            self.sigm0 /= 2. * np.pi * np.power(self.rval0 * sc.au * 1e2, 2)
            self.sigm0 *= np.exp(np.power(self.rvals[0] / self.rval0, q))
            self.sigma = self._calc_surfacedensity()
        else:
            q = (2. - self.sigmq)
            self.mdisk = self.sigm0 / q / self.msun / 1e3
            self.mdisk *= 2. * np.pi * np.power(self.rval0 * sc.au * 1e2, 2)
            self.mdisk *= np.exp(np.power(self.rvals[0] / self.rval0, q))
            self.sigma = self._calc_surfacedensity()
        self.mdisk = self._calc_mdisk()

        # Include the gap perturbations. This is similar to the f(R) value in
        # Eqn. 12 of van Boekel et al. (2017). If no gaps are specified, this
        # is simply an array of ones.

        self.gaps = kwargs.get('gaps', kwargs.get('gap', None))
        self.depletion_profile = self._calc_depletion_profile(self.gaps)
        self.sigma = self.sigma * self.depletion_profile
        self.mdisk = self._calc_mdisk()

        # Temperature - two layer approximation from Dartois et al. (2013).
        # The {Htmp0, Htmpq} values describe the transition between the
        # midplane and atmospheric temperature regime. By default, this will
        # use Zq = 4Hp, where Hp is the pressure scale height.

        self.tmid0 = kwargs.get('tmid0', 20.)
        self.tatm0 = kwargs.get('tatm0', 90.)
        self.tmidq = kwargs.get('tmidq', -0.3)
        self.tatmq = kwargs.get('tatmq', -0.3)
        self.delta = kwargs.get('delta', 2.0)
        self.Htmp0 = kwargs.get('Htmp0', None)
        self.Htmpq = kwargs.get('Htmpq', None)

        # Volume density - by default the vertical structure is hydrostatic.
        # The {Hdns0, Hdnsq} parameters descibe the density scale height. If
        # dfunc='gaussian' then this will set the scaleheight used for that.
        # If none are specified, the pressure scale height is used by deafult.
        # A minimum density can be set, useful for trimming large models.

        self.Hdns0 = kwargs.get('Hdns0', None)
        self.Hdnsq = kwargs.get('Hdnsq', None)
        self.minH2 = kwargs.get('minH2', 1.e3)

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

        # CO distribution. By default assume a disk-wide abundance value.

        self.xgas = self._make_profile(kwargs.get('xgas', 1e-4))
        self.xdep = self._make_profile(kwargs.get('xdep', self.xgas * 1e-6))
        self.column = kwargs.get('column', None)
        self.ndiss = kwargs.get('ndiss', 1.3e21)
        self.tfreeze = kwargs.get('tfreeze', 19.)

        if self.column is not None:
            self.column = self._make_profile(self.column)
            if self.verbose and all(self.xgas != 1e-4):
                print("Warning: column density will overwrite abundance.")

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
        self.incl_pressure = kwargs.get('incl_pressure', True)
        self.incl_diskmass = kwargs.get('incl_diskmass', False)
        self.rotation = self._calc_rotation()
        return

    def _gaussian_gap(self, gap):
        """Return a Gaussian shaped gap. Can include variable width."""
        try:
            dx1, dx2 = gap[1]
        except:
            dx1 = gap[1]
            dx2 = dx1
        gap1 = self._gaussian(self.rvals, gap[0], dx1, gap[2])
        gap2 = self._gaussian(self.rvals, gap[0], dx2, gap[2])
        return 1. - np.where(self.rvals <= gap[0], gap1, gap2)

    def _square_gap(self, gap):
        """Return a square edged gap."""
        return np.where(abs(self.rvals - gap[0]) / 2.0 < gap[1], gap[2], 1.)

    def _duffell_gap(self, gap):
        """Return a gap with profile described in Duffel (2015)."""
        raise NotImplementedError("Choose 'gaussian' or 'square'.")

    def _calc_depletion_gap(self, gap):
        """Return the depletion profile for a single gap."""
        if gap[-1] == 'square':
            return self._square_gap(gap)
        elif gap[-1] == 'duffell':
            return self._duffell_gap(gap)
        else:
            return self._gaussian_gap(gap)

    def _calc_depletion_profile(self, gaps):
        """Calculates the combined perturbing profile for all gap(s)."""
        if gaps is None:
            return np.ones(self.nr)
        try:
            profile = [self._calc_depletion_gap(gap) for gap in gaps]
            profile = np.product(profile, axis=0)
        except IndexError:
            profile = self._calc_depletion_gap(gaps)
        return profile

    def _calc_rotation(self):
        """Calculate the azimuthal rotation velocity [m/s]."""

        # Keplerian rotation.
        if self.incl_altitude:
            vkep = np.hypot(self.rvals_m[None, :], self.zvals_m[:, None])
        else:
            vkep = self.rvals_m[None, :] * np.ones(self.nz)[:, None]
        vkep = np.power(self.rvals_m, 2)[None, :] * np.power(vkep, -3)
        vkep *= sc.G * self.mstar * self.msun

        # Disk gravity component.
        if self.incl_diskmass:
            if type(self.incl_diskmass) is not bool:
                N = self.incl_diskmass
            else:
                N = 10
            self.phi = self._calc_gravitational_potential(N)
            dphidr = np.gradient(self.phi, self.rvals_m, axis=1)
            vgrav = self.rvals_m[None, :] * dphidr
            vgrav = np.where(np.isfinite(vgrav), vgrav, 0.0)
        else:
            vgrav = 0.0

        # Pressure gradient.
        if self.incl_pressure:
            n = self.dens * 1e6
            P = n * sc.k * self.temp
            dPdr = np.gradient(P, self.rvals_m, axis=1)
            vpres = self.rvals_m[None, :] * dPdr / n / sc.m_p / self.mu
            vpres = np.where(np.isfinite(vpres), vpres, 0.0)
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
        sigma = np.where(np.isfinite(self.sigma), self.sigma, 0.0)
        mdisk = np.trapz(sigma * self.rvals_cm, self.rvals_cm)
        return 2. * np.pi * mdisk / 1e3 / self.msun

    def _calc_gravitational_potential(self, N=5):
        """Calculate the gravitational potential [m^2/s^2]."""

        # Build new axes as we need to solve this in 3D.
        # Grid is in units of [m].

        xvals = np.linspace(-self.rmax, self.rmax, int(self.nr / 10)) * sc.au
        yvals = np.linspace(-self.rmax, self.rmax, int(self.nr / 10)) * sc.au
        zvals = np.linspace(-self.zmax, self.zmax, int(self.nz / 10)) * sc.au

        cell_volume = np.diff(xvals)[0]**2 * np.diff(zvals)[0] * 1e6
        grid_shape = (zvals.size, yvals.size, xvals.size)
        xvals = xvals[None, None, :] * np.ones(grid_shape)
        yvals = yvals[None, :, None] * np.ones(grid_shape)
        zvals = zvals[:, None, None] * np.ones(grid_shape)
        rvals = np.hypot(xvals, yvals)

        # Interpolate the mass points onto the disk.

        mass = (self.dens * cell_volume * self.mu * sc.m_p).flatten()
        mass = griddata((self.rpnts.flatten(), self.zpnts.flatten()), mass,
                        (rvals.flatten() / sc.au,
                         abs(zvals).flatten() / sc.au),
                        method='linear').reshape(rvals.shape)

        # Loop through the points.

        r_sample, z_sample = self.rvals_m[::N], self.zvals_m[::N]
        potential = np.zeros((z_sample.size, r_sample.size))

        for zidx, alt in enumerate(z_sample):
            for ridx, rad in enumerate(r_sample):
                dx = np.power(xvals - rad, 2.0)
                dy = np.power(yvals, 2.0)
                dz = np.power(zvals - alt, 2.0)
                dr = np.sqrt(dx + dy + dz)
                dr = np.where(dr == 0.0, 1e50, dr)
                potential[zidx, ridx] = -sc.G * np.nansum(mass / dr)

        if N > 1:

            # Linearly interpolate the potential. For the gridding we use both
            # linear and nearest methods to account for the regions near the
            # edge of the grid.

            rpnts, zpnts = np.meshgrid(r_sample / sc.au, z_sample / sc.au)
            rpnts, zpnts = rpnts.flatten(), zpnts.flatten()
            linear = griddata((rpnts, zpnts), potential.flatten(),
                              (self.rvals[None, :], self.zvals[:, None]),
                              method='cubic', fill_value=np.nan)
            nearest = griddata((rpnts, zpnts), potential.flatten(),
                               (self.rvals[None, :], self.zvals[:, None]),
                               method='nearest')
            potential = np.where(np.isfinite(linear), linear, nearest)

        return potential

    def radialpowerlaw(self, x0, q):
        """Radial power law."""
        return x0 * np.power(self.rvals / self.rval0, q)

    def _calc_surfacedensity(self):
        """Surface density in [g / cm^2]."""
        sigm = self.radialpowerlaw(self.sigm0, -self.sigmq)
        sigm *= np.exp(self.radialpowerlaw(-1.0, 2.-self.sigmq))
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
                print("Reverting to Zq = 4Hp.")
        if self.Htmp0 is not None and self.Htmpq is not None:
            zq = self.radialpowerlaw(self.Htmp0, self.Htmpq)
        else:
            zq = 4 * self._calc_scaleheight(tmid=tmid)
        T = np.cos(np.pi * self.zvals[:, None] / 2. / zq[None, :])
        T = np.power(T, 2 * self.delta)
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
            sigma = self.sigma
        rho *= sigma[None, :] / np.trapz(rho, self.zvals_cm, axis=0)
        rho /= self.mu * sc.m_p * 2e3
        mask = np.logical_and(np.isfinite(rho), rho >= self.minH2)
        return np.where(mask, rho, 0.0)

    def _calc_volumedensity_hydrostatic(self, temp=None, sigma=None):
        """Hydrostatic density distribution [/cm^3]."""
        if temp is None:
            temp = self.temp
        if sigma is None:
            sigma = self.sigma
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
            sigma = self.sigma
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

        # If a column density is specified, calculate the appropriate x_gas
        # value, assuming that the x_dep value still holds.

        if self.column is not None:
            NH2_gas = np.nansum(dens * mask * dz, axis=0)
            NH2_dep = self.sigma / sc.m_p / self.mu / 1e3 - NH2_gas
            self.xgas = self.column / (NH2_gas + 1e-4 * NH2_dep)
            self.xdep = 1e-4 * self.xgas
        return np.where(mask, self.xgas[None, :], self.xdep[None, :])

    def _sample_input(self, arr_y):
        """Sample 'arr' across the model radius."""
        arr_x = np.linspace(self.rmin, self.rmax, len(arr_y))
        return np.interp(self.rvals, arr_x, arr_y)

    def _make_profile(self, value):
        """Make the input into a radial profile."""
        if type(value) in [float, np.float32, np.float64, np.float128]:
            return np.array([value for _ in range(self.nr)])
        return self._sample_input(value)

    # Functions to do simple radiative transfer.

    def _calc_phi(self):
        """Return the FWHM of the line at each cell."""
        dV = 2. * sc.k * self.temp / self.rates.mu / sc.m_p
        dV = np.sqrt(dV + self.vturb**2)
        return 2. * np.sqrt(np.log(2.)) * dV

    def _calc_Qrot(self, T):
        """Rotational parition function."""
        return sc.k * T / sc.h / self.B0 + 1. / 3.

    def _level_population(self, J=0):
        """Return the densities of the provided level."""
        n = self.dens * self.abun
        n *= self.rates.levels[J+1].g / self._calc_Qrot(self.temp)
        return n / np.exp(self.rates.lines[J+1].Eup / self.temp)

    def _absorption_coefficient(self, J=0):
        """Return the asborption coefficient for the (J+1 - J) level."""
        n1 = self._level_population(J)
        g2 = self.rates.levels[J+2].g
        g1 = self.rates.levels[J+1].g
        A21 = self.rates.lines[J+1].A
        nu = self.rates.lines[J+1].freq
        phi = self._calc_phi()
        alpha = A21 * sc.c**2 * n1 * g2 * phi / 8. / np.pi / nu**2 / g1
        return alpha * (1. - np.exp(-sc.h * nu / sc.k / self.temp))

    def _calc_tau(self, J=0):
        """Return the integrated optical depth of each cell."""
        alpha = self._absorption_coefficient(J)
        dz = np.diff(self.zvals_cm)
        dz = np.insert(dz, 0, dz[0])
        return alpha * dz

    def _calc_source_function(self, J=0):
        """Calculate the source function for each cell."""
        nu = self.rates.lines[J+1].freq
        Snu = 2. * sc.h * nu**3 / sc.c**2
        return Snu / (np.exp(sc.h * nu / sc.k / self.temp) - 1.)

    def _calc_intensity_weights(self, J=0):
        """Emission weights for each cell."""

        Snu = self._calc_source_function(J)
        tau = self._calc_tau(J)

        zidx = int(self.zmin == 0.0)
        tau = np.vstack([np.flipud(tau[zidx:]), tau])
        Snu = np.vstack([np.flipud(Snu[zidx:]), Snu])
        ctau = np.cumsum(tau[::-1], axis=0)[::-1]

        intensity = Snu * (1. - np.exp(-tau)) * np.exp(-ctau)
        intensity /= np.nansum(intensity, axis=0)[None, :]
        return np.where(np.log(intensity) < -20., 0.0, intensity)

    def _calc_weighted_parameter(self, param, J=0, absolute=False):
        """Return the weighted of the parameter."""

        zidx = int(self.zmin == 0.0)
        if param.shape == self.dens.shape:
            param = np.vstack([np.flipud(param[zidx:]), param])
        elif param.shape == self.zvals.shape:
            param = param[:, None] * np.ones((self.nz, self.nr))
            param = np.vstack([-np.flipud(param[zidx:]), param])
        else:
            raise ValueError("Unknown shape for 'param'.")
        if absolute:
            param = abs(param)

        weights = self._calc_intensity_weights(J)
        if weights.shape[0] != param.shape[0]:
            raise ValueError("Mirroring did not work.")

        pcnts = np.zeros((self.nr, 3))
        for ridx in range(self.nr):
            pcnts[ridx] = self._wpcnts(param[:, ridx], weights[:, ridx])
        return pcnts.T

    def emission_temperature(self, J=0):
        """Return the temperature traced by emission."""
        return self._calc_weighted_parameter(self.temp, J)

    def emission_rotation(self, J=0):
        """Return the rotation traced by emission."""
        return self._calc_weighted_parameter(self.rotation, J)

    def emission_height(self, J=0, absolute=True):
        """Return the rotation traced by emission."""
        return self._calc_weighted_parameter(self.zvals, J, absolute)

    def emission_density(self, J=0):
        """Return the rotation traced by emission."""
        return self._calc_weighted_parameter(self.dens, J)

    def _wpcnts(self, data, weights, percentiles=[0.16, 0.5, 0.84]):
        '''Weighted percentiles.'''
        idx = np.argsort(data)
        sorted_data = np.take(data, idx)
        sorted_weights = np.take(weights, idx)
        cum_weights = np.add.accumulate(sorted_weights)
        scaled_weights = (cum_weights - 0.5 * sorted_weights) / cum_weights[-1]
        spots = np.searchsorted(scaled_weights, percentiles)
        wp = []
        for s, p in zip(spots, percentiles):
            if s == 0:
                wp.append(sorted_data[s])
            elif s == data.size:
                wp.append(sorted_data[s-1])
            else:
                f1 = (scaled_weights[s] - p)
                f1 /= (scaled_weights[s] - scaled_weights[s-1])
                f2 = (p - scaled_weights[s-1])
                f2 /= (scaled_weights[s] - scaled_weights[s-1])
                wp.append(sorted_data[s-1] * f1 + sorted_data[s] * f2)
        return np.array(wp)

    # Functions to write the model out to limepy or similar.

    def write_header(self, filename, minN=0.0, minH2=0.0):
        """Write the model to a .h file for LIME."""

        rpnts = self.rvals[None, :] * np.ones(self.nz)[:, None]
        zpnts = np.ones(self.nr)[None, :] * self.zvals[:, None]
        Npnts = self.column[None, :] * np.ones(self.nz)[:, None]
        rpnts = rpnts.T.flatten()
        zpnts = zpnts.T.flatten()
        Npnts = Npnts.T.flatten()
        dflat = self.dens.T.flatten()
        tflat = self.temp.T.flatten()
        aflat = self.abun.T.flatten()
        vflat = self.rotation.T.flatten()

        # Generate a mask to remove bad points.

        mask = np.ones(rpnts.size)
        mask *= np.logical_and(dflat > minH2, Npnts > minN)
        mask *= np.logical_and(np.isfinite(tflat), tflat > 0.0)
        mask *= np.logical_and(rpnts > 0, zpnts >= 0)
        mask *= np.logical_and(vflat != np.inf, np.isfinite(vflat))
        mask = mask.astype('bool')

        rpnts = rpnts[mask]
        zpnts = zpnts[mask]
        dflat = dflat[mask]
        tflat = tflat[mask]
        aflat = aflat[mask]
        vflat = vflat[mask]

        # Change to LIME appropriate units.
        dflat *= 1e6

        # Write the strings to a single file.
        arrays = np.nan_to_num([rpnts, zpnts, dflat, tflat, aflat, vflat])
        arraynames = ['c1arr', 'c2arr', 'dens', 'temp', 'abund', 'vrot']
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

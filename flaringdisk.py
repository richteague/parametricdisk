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

import numpy as np
import scipy.constants as sc


class ppd:

    msun = 1.988e30     # Stellar mass.
    mu = 2.35           # Mean molecular weight.

    def __init__(self, **kwargs):
        """Protoplanetary disk model."""

        self.mstar = kwargs.get('mstar', 1.0)
        self.rval0 = kwargs.get('rval0', 200.)
        self.sigm0 = kwargs.get('sigm0', 15.)
        self.sigmq = kwargs.get('sigmq', -1.)
        self.tmid0 = kwargs.get('tmid0', 20.)
        self.tatm0 = kwargs.get('tatm0', 90.)
        self.tmidq = kwargs.get('tmidq', -0.3)
        self.tatmq = kwargs.get('tatmq', -0.3)
        self.Htmp0 = kwargs.get('Htmp0', None)
        self.Htmpq = kwargs.get('Htmpq', None)
        self.Hdns0 = kwargs.get('Hdns0', None)
        self.Hdnsq = kwargs.get('Hdnsq', None)
        self.dfunc = kwargs.get('dfunc', 'hydrostatic').lower()
        self.ndiss = kwargs.get('ndiss', 1.3e21)
        self.tfreeze = kwargs.get('tfreeze', 20.)
        self.xgas = kwargs.get('xgas', 1e-4)
        self.xdep = kwargs.get('xdep', 1e-12)
        self.rin = kwargs.get('rin', 0.0)

        # Select the correct density functional form.

        if self.dfunc not in ['gaussian', 'hydrostatic']:
            raise ValueError("dfunc must be 'gaussian' or 'hydrostatic'.")
        else:
            if self.dfunc == 'gaussian':
                self.volumedensity = self.volumedensity_gaussian
            else:
                self.volumedensity = self.volumedensity_hydrostatic

        # Check the values describing the flaring.

        flag = False
        if all([self.Htmp0 is not None, self.Htmpq is not None]):
            print("Will used modified temperature structure.")
            flag = True
        if all([self.Hdns0 is not None, self.Hdnsq is not None]):
            if flag:
                raise ValueError("All four scale height parameters set.")
            if self.dfunc == 'gaussian':
                print("Will use user-controlled density scale heights.")
            else:
                print("Must set 'dfunc' to 'gaussian' to use 'Hdns' params.")

        # The grids, by default, are functions of the characteristic radius.

        self.nrpnts = kwargs.get('nr', 100)
        self.nzpnts = kwargs.get('nz', 100)
        self.rvals = kwargs.get('rvals', np.linspace(0.1, 5, self.nrpnts))
        self.rvals *= self.rval0
        self.zvals = kwargs.get('zvals', np.linspace(0, 5, self.nzpnts))
        self.zvals *= self.rval0

        # Build the disk model.
        self.sigm = self.surfacedensity()
        self.tmid = self.midplanetemp()
        self.tatm = self.atmospheretemp()
        self.temp = self.temperature()
        self.dens = self.volumedensity()
        self.Hp = self.scaleheight()
        self.abun = self.abundance()
        return

    def radialpowerlaw(self, x0, q):
        """Radial power law."""
        return x0 * np.power(self.rvals / self.rval0, q)

    def surfacedensity(self):
        """Surface density in [g / cm^2]."""
        sigm = self.radialpowerlaw(self.sigm0, self.sigmq)
        sigm *= np.exp(self.radialpowerlaw(-1.0, 2.+self.sigmq))
        return np.where(self.rvals >= self.rin, sigm, 0.0)

    def midplanetemp(self):
        """Midplane temperature in [K]."""
        return self.radialpowerlaw(self.tmid0, self.tmidq)

    def atmospheretemp(self):
        """Atmospheric temperature in [K]."""
        return self.radialpowerlaw(self.tatm0, self.tatmq)

    def scaleheight(self, tmid=None):
        """Pressure scale height in [au]."""
        if self.dfunc == 'gaussian':
            if self.Hdns0 is not None and self.Hdnsq is not None:
                return self.radialpowerlaw(self.Hdns0, self.Hdnsq)
            elif any([self.Hdns0 is not None, self.Hdnsq is not None]):
                print("Both 'Hdns0' and 'Hdnsq' must be set.")
                print("Reverting to pressure scale height.")
        if tmid is None:
            tmid = self.midplanetemp()
        Hp = sc.k * tmid * np.power(self.rvals, 3) / self.mu / sc.m_p
        return np.sqrt(Hp * sc.au / sc.G / self.mstar / self.msun)

    def temperature(self, tmid=None, tatm=None):
        """Gas temperature in [K]."""
        if tmid is None:
            tmid = self.tmid
        if tatm is None:
            tatm = self.tatm
        if all([self.Htmp0 is not None, self.Htmpq is not None]):
            zq = self.radialpowerlaw(self.Htmp0, self.Htmpq)
        else:
            zq = 4 * self.scaleheight(tmid=tmid)
        if any([self.Htmp0 is not None, self.Htmpq is not None]):
            print("Both 'Htmp0' and 'Htmpq' must be set.")
            print("Reverting to Zq is four pressure scale heights.")
        temp = np.sin(np.pi * self.zvals[:, None] / 2. / zq[None, :])**2
        temp = temp * (tatm - tmid)[None, :] + tmid[None, :]
        return np.where(self.zvals[:, None] > zq[None, :], tatm[None, :], temp)

    def soundspeed(self, temp=None):
        """Sound speed in [m/s]."""
        if temp is None:
            temp = self.temperature()
        return np.sqrt(sc.k * temp / self.mu / sc.m_p)

    def volumedensity_normalise(self, rho, sigma=None):
        """Normalised the density profile."""
        if sigma is None:
            sigma = self.sigm
        rho *= sigma[None, :] / np.trapz(rho, self.zvals * sc.au * 100, axis=0)
        rho /= self.mu * sc.m_p * 1e3
        return np.where(np.logical_and(np.isfinite(rho), rho >= 0.0), rho, 0.0)

    def volumedensity_hydrostatic(self, temp=None, sigma=None):
        """Hydrostatic density distribution [/cm^3]."""
        if temp is None:
            temp = self.temp
        if sigma is None:
            sigma = self.sigm
        dT = np.diff(np.log(temp), axis=0)
        dT = np.vstack((dT, dT[-1]))
        dz = np.diff(self.zvals * sc.au)
        dz = np.hstack((dz, dz[-1]))
        cs = dz[:, None] * np.power(self.soundspeed(temp), -2.)[None, :]
        G = (np.hypot(self.rvals[None, :], self.zvals[:, None]) * sc.au)**3
        G = sc.G * self.mstar * self.msun * self.zvals[:, None] * sc.au / G
        drho = np.squeeze(dT + cs * G)
        rho = np.ones(temp.shape)
        for i in range(self.rvals.size):
            for j in range(1, self.zvals.size):
                rho[j, i] = np.exp(np.log(rho[j-1, i]) - drho[j, i])
        return self.volumedensity_normalise(rho, sigma)

    def volumedensity_gaussian(self, temp=None, sigma=None):
        """Gaussian density distribution [/cm^3]."""
        if temp is None:
            temp = self.temp
        if sigma is None:
            sigma = self.sigm
        Hp = self.scaleheight(temp[abs(self.zvals).argmin()])
        rho = np.exp(-0.5 * (self.zvals[:, None] / Hp[None, :])**2)
        rho /= np.sqrt(2. * np.pi) * Hp * sc.au * 100.
        return self.volumedensity_normalise(rho, sigma)

    def abundance(self, temp=None, dens=None):
        """Returns molecular abundance [wrt H2]."""
        if temp is None:
            temp = self.temp
        if dens is None:
            dens = self.dens
        dz = abs(np.diff(self.zvals).mean()) * sc.au * 1e2
        col = np.cumsum(dens[::-1], axis=0)[::-1] * dz
        abun = np.where(col > self.ndiss, self.xgas, self.xdep)
        return np.where(temp > self.tfreeze, abun, self.xdep)

    # Functions to write the model out to limepy or similar.

    def flatten(self, arr):
        """Flatten array in way required for limepy."""
        return arr.swapaxes(1, 2).flatten()

    def write_header(self, filename, clipdens=1e3, resample=1, retvals=False):
        """Write the model to a .h file for LIME."""

        # Broadcast the points.
        rpntflat = self.rvals[None, None, :] * np.ones(self.dens.shape)
        zpntflat = self.zvals[None, :, None] * np.ones(self.dens.shape)
        tpntflat = self.tvals[:, None, None] * np.ones(self.dens.shape)

        # Flatten the arrays.
        rpntflat = self.flatten(rpntflat)
        zpntflat = self.flatten(zpntflat)
        tpntflat = self.flatten(tpntflat)
        densflat = self.flatten(self.dens)
        tempflat = self.flatten(self.temp)
        abunflat = self.flatten(self.abun)
        print('Model has %d points.' % len(rpntflat))

        # Remove low density points.
        if clipdens is not None:
            print('Removing low density points...')
            mask = np.array([d >= clipdens for d in densflat])
            rpntflat = rpntflat[mask]
            zpntflat = zpntflat[mask]
            tpntflat = tpntflat[mask]
            densflat = densflat[mask]
            tempflat = tempflat[mask]
            abunflat = abunflat[mask]
            print('%d points remain.' % len(rpntflat))

        # Removing points outside the allowed theta range (-pi, pi).
        if zpntflat.min() < np.pi:
            print('Removing theta values less than negative pi.')
            mask = np.array([t > -np.pi for t in tpntflat])
            rpntflat = rpntflat[mask]
            zpntflat = zpntflat[mask]
            tpntflat = tpntflat[mask]
            densflat = densflat[mask]
            tempflat = tempflat[mask]
            abunflat = abunflat[mask]
            print('%d points remain.' % len(rpntflat))

        if zpntflat.max() > np.pi:
            print('Removing theta values less than negative pi.')
            mask = np.array([t < np.pi for t in tpntflat])
            rpntflat = rpntflat[mask]
            zpntflat = zpntflat[mask]
            tpntflat = tpntflat[mask]
            densflat = densflat[mask]
            tempflat = tempflat[mask]
            abunflat = abunflat[mask]
            print('%d points remain.' % len(rpntflat))

        # Resample theta if appropriate.
        if resample is not None and resample > 1:
            print('Resample the theta axis...')
            tokeep = self.tvals[::int(resample)]
            mask = np.array([t in tokeep for t in tpntflat])
            rpntflat = rpntflat[mask]
            zpntflat = zpntflat[mask]
            tpntflat = tpntflat[mask]
            densflat = densflat[mask]
            tempflat = tempflat[mask]
            abunflat = abunflat[mask]
            print('%d points remain.' % len(rpntflat))

        # Convert to LIME specific units.
        densflat *= 1e6

        # Write to the header file.
        arrays = [rpntflat, zpntflat, tpntflat, densflat, tempflat, abunflat]
        anames = ['c1arr', 'c2arr', 'c3arr', 'dens', 'temp', 'abund']
        string = ''
        for a, n in zip(arrays, anames):
            string += self._write_header_string(a, n)
        with open('%s.h' % filename.replace('.h', ''), 'w') as hfile:
            hfile.write('%s' % string)
        print 'Written to %s.h' % filename.replace('.h', '')
        if retvals:
            return np.squeeze(arrays)
        return

    def _write_header_string(self, array, name):
        """Returns a string of the array to save."""
        tosave = 'const static double %s[%d] = {' % (name, array.size)
        for val in array:
            tosave += '%.3e, ' % val
        tosave = tosave[:-2] + '};\n'
        return tosave

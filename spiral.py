"""
A protoplanetary disk with a perturbation in the midplane temperature, the
surface density or both. The overall model is a reimplementation of the
Williams and Best (2014) model. The perturbation is from Juhasz et al (2015)
which uses the wake equation of Rafikov (2002).
"""

import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d


class ppd:

    msun = 1.988e30     # Stellar mass.
    mu = 2.35           # Mean molecular weight.

    def __init__(self, **kwargs):
        """Protoplanetary disk with a spiral perturbation."""

        # Star, planet and disk values. By default the reference radius is the
        # planetary radius.

        self.mstar = kwargs.get('mstar', 1.0)
        self.mplanet = kwargs.get('mplanet', 1e-3)
        self.rplanet = kwargs.get('rplanet', 10.)
        self.tplanet = kwargs.get('tplanet', 0.0)
        self.rval0 = kwargs.get('rval0', self.rplanet)
        self.sigm0 = kwargs.get('sigm0', 15.)
        self.sigmq = kwargs.get('sigmq', -1.)
        self.tmid0 = kwargs.get('tmid0', 20.)
        self.tatm0 = kwargs.get('tatm0', 100.)
        self.tmidq = kwargs.get('tmidq', -0.4)
        self.tatmq = kwargs.get('tatmq', -0.4)
        self.nHpZq = kwargs.get('nHpZq', 4.)
        self.dfunc = kwargs.get('dfunc', 'gaussian').lower()

        # Select the correct density functional form.

        if self.dfunc not in ['gaussian', 'hydrostatic']:
            raise ValueError("dfunc must be 'gaussian' or 'hydrostatic'.")
        else:
            if self.dfunc == 'gaussian':
                self.volumedensity = self.volumedensity_gaussian
            else:
                self.volumedensity = self.volumedensity_hydrostatic

        # The grids, by default, are functions of the planet radius.

        self.rvals = kwargs.get('rvals',
                                np.linspace(0.2, 5, 200) * self.rplanet)
        self.zvals = kwargs.get('zvals',
                                np.linspace(0, 3.5, 100) * self.rplanet)
        self.tvals = kwargs.get('tvals',
                                np.linspace(-np.pi, np.pi, 360))

        # Properties to define the wake, see Juhasz et al. (2015) for more.

        self.dT = kwargs.get('dT', 0.0)
        self.dS = kwargs.get('dS', 0.0)
        self.waked = kwargs.get('waked', 0.8)
        self.wakeq = kwargs.get('wakeq', -1.7)
        self.alpha = 1.5
        self.beta = -0.5 * self.tmidq
        self.wakeradius = interp1d(self.waketheta(), self.rvals,
                                   bounds_error=False)
        self.nwake = np.ceil(max(abs(self.wakeradius.x.max() / 2. / np.pi),
                             self.wakeradius.x.max() / 2. / np.pi))

        # Build the disk model.
        self.pert = np.squeeze([[self.perturbation(r, t) for r in self.rvals]
                                for t in self.tvals])
        self.sigm = np.squeeze([self.surfacedensity(self.dS * p)
                                for p in self.pert])
        self.tmid = np.squeeze([self.midplanetemp(self.dT * p)
                                for p in self.pert])
        self.temp = np.squeeze([self.temperature(t)
                                for t in self.tmid])
        self.dens = np.squeeze([self.volumedensity(t, s)
                                for t, s in zip(self.temp, self.sigm)])
        self.Hp = np.squeeze([self.scaleheight(t) for t in self.temp])
        return

    def radialpowerlaw(self, x0, q):
        """Radial power law."""
        return x0 * np.power(self.rvals / self.rval0, q)

    def surfacedensity(self, pert=0.0):
        """Surface density in [g / cm^2]."""
        return self.radialpowerlaw(self.sigm0, self.sigmq) * (1.0 + pert)

    def midplanetemp(self, pert=0.0):
        """Midplane temperature in [K]."""
        return self.radialpowerlaw(self.tmid0, self.tmidq) * (1.0 + pert)

    def atmospheretemp(self):
        """Atmospheric temperature in [K]."""
        return self.radialpowerlaw(self.tatm0, self.tatmq)

    def scaleheight(self, tmid=None):
        """Pressure scale height in [au]."""
        if tmid is None:
            tmid = self.midplanetemp()
        Hp = sc.k * tmid * np.power(self.rvals, 3) / self.mu / sc.m_p
        return np.sqrt(Hp * sc.au / sc.G / self.mstar / self.msun)

    def temperature(self, tmid=None):
        """Gas temperature in [K]."""
        if tmid is None:
            tmid = self.midplanetemp()
        tatm = self.atmospheretemp()
        zq = self.nHpZq * self.scaleheight(tmid)
        temp = np.sin(np.pi * self.zvals[:, None] / 2. / zq[None, :])**2
        temp = temp * (tatm - tmid)[None, :] + tmid[None, :]
        return np.where(self.zvals[:, None] > zq[None, :], tatm[None, :], temp)

    def soundspeed(self, temp=None):
        """Sound speed in [m/s]."""
        if temp is None:
            temp = self.temperature()
        return np.sqrt(sc.k * temp / self.mu / sc.m_p)

    def volumedensity_normalise(self, rho, sigma):
        """Normalised the density profile."""
        rho *= sigma[None, :] / np.trapz(rho, self.zvals * sc.au * 100, axis=0)
        rho /= self.mu * sc.m_p * 1e3
        return np.where(np.logical_and(np.isfinite(rho), rho >= 0.0), rho, 0.0)

    def volumedensity_hydrostatic(self, temp, sigma):
        """Hydrostatic density distribution [/cm^3]."""
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

    def volumedensity_gaussian(self, temp, sigma):
        """Gaussian density distribution [/cm^3]."""
        Hp = self.scaleheight(temp[abs(self.zvals).argmin()])
        rho = np.exp(-0.5 * (self.zvals[:, None] / Hp[None, :])**2)
        rho /= np.sqrt(2. * np.pi) * Hp * sc.au * 100.
        return self.volumedensity_normalise(rho, sigma)

    def abundance(self, dens, temp):
        """Returns if molecules can survive."""

        return

    def perturbation(self, r, t):
        """Perturbation function."""
        dA = np.power(r / self.rplanet, np.sign(r - self.rplanet) * self.wakeq)
        dA *= np.exp(-1 * np.power(self.wakedistance(r, t) / self.waked, 2))
        return dA

    def waketheta(self):
        """Position angle of the wake at radius r."""
        sgn = np.sign(self.rvals - self.rplanet)
        a = sgn * self.rvals / self.scaleheight()
        b = np.power(self.rvals / self.rplanet, -self.alpha)
        b /= (1. - self.alpha + self.beta)
        b = 1. / (1. + self.beta) - b
        c = 1. / (1. + self.beta)
        c -= 1. / (1. - self.alpha + self.beta)
        return self.tplanet - a * (self.radialpowerlaw(b, 1 + self.beta) - c)

    def wakedistance(self, r, t):
        """Radial distance [au] to the nearest wake."""
        rwake = np.array([self.wakeradius(t + i * np.pi * 2)
                          for i in np.arange(-self.nwake, self.nwake)])
        return np.nanmin(abs(rwake - r))

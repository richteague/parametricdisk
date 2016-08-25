# Functions to create a parametric protoplanetary disk model.
# Based mainly on Rosenfeld et al. (2013) and Flaherty et al. (2015).
# Focusses on the gas. Dust can be scaled appropriately.

import numpy as np
import scipy.constants as sc


# Pressure scale height based on midplane temperature.
def getScaleHeight(r, q_mid, T_mid, mu=2.37, Mstar=0.6):
    Tm = T_mid * np.power(r / 10., -q_mid)
    Hp = sc.k * Tm * r**3 * sc.au / mu / sc.m_p / sc.G / Mstar / 1.99e30
    return np.sqrt(Hp)


# Gas temperature from two power-laws.
def getTemperature(r, z, q_mid, q_atm, T_mid, T_atm, mu=2.37):
    Ta = getTemperaturePL(r, q_atm, T_atm)
    Tm = getTemperaturePL(r, q_mid, T_mid)
    zq = 4. * getScaleHeight(r, q_mid, T_mid, mu=mu)
    T = np.where(z > zq, Ta, Ta + (Tm - Ta) * np.cos(np.pi * z / 2. / zq)**2.)
    return T


# Get the temperature power law.
def getTemperaturePL(r, q, T):
    return T * np.power(r / 10., -q)


# Surface density profile.
def getSurfaceDensity(r, sigma_c, r_c, gamma):
    sigma = sigma_c * np.power(r/r_c, -gamma)
    return sigma * np.exp(-1. * np.power(r/r_c, 2-gamma))


# The sound speed of the gas.
def getSoundSpeed(temp, mu=2.37):
    return np.sqrt(sc.k * temp / mu / sc.m_p)


# Density structure assuming hydrostatic equilibrium.
def getNonIsothermalDensity(rvals, zvals, temp, sigma, Mstar=0.6):
    if temp.ndim != 2:
        raise ValueError("temp must be 2D.")
    elif temp.shape != (zvals.size, rvals.size):
        raise ValueError("temp must be on (rvals, zvals) grid.")
    if rvals.size != sigma.size:
        raise ValueError("sigma must be sampled at rvals.")
    dens = np.ones((zvals.size, rvals.size))
    for i in range(1, int(zvals.size)):
        dlnT = np.log(temp[i-1] / temp[i])
        dlnT = np.where(np.isfinite(dlnT), dlnT, 0.0)
        dzdc = (zvals[i-1] - zvals[i]) * sc.au
        dzdc /= getSoundSpeed(temp[i])**2.
        dzdc = np.where(np.isfinite(dzdc), dzdc, 0.0)
        grav = sc.G * Mstar * 1e30 * zvals[i] * sc.au
        grav /= np.hypot(rvals * sc.au, zvals[i] * sc.au)**3.
        grav = np.where(np.isfinite(grav), grav, 0.0)
        dens[i] = dens[i-1] * np.exp(dlnT + dzdc*grav)
    for i in range(rvals.size):
        dens[:, i] *= sigma[i] / np.trapz(dens[:, i], x=zvals*sc.au*100.)
    return dens


# Density structure assuming vertically isothermal disks.
def getIsothermalDensity(zvals, sigma, scaleheight):
    dens = np.power(zvals[:, None], 2) / 2. / np.power(scaleheight[None, :], 2)
    dens = np.exp(-dens)
    dens *= sigma / scaleheight / sc.au / 100. / np.sqrt(2. * np.pi)
    return dens


# Get the CO parameterised CO abundance.
def getCOAbundance(ygrid, dens, temp, n_gas=1e-4, n_ice=0.0, T_freeze=20., 
                   N_diss=1.3e21):
    dens = np.where(np.isfinite(dens), dens, 0.0)
    temp = np.where(np.isfinite(temp), temp, 0.0)
    N_H2 = np.array([np.trapz(dens[i:], x=ygrid[i:]*sc.au*100., axis=0) 
                     for i in range(ygrid.size)])
    return np.where(np.logical_and(temp > T_freeze, N_H2 > N_diss), 
                    n_gas*dens, n_ice*dens)

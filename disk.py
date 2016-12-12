import numpy as np
import scipy.constants as sc
from scipy.ndimage.interpolation import map_coordinates

class ppd:
    
    def __init__(self, r_c, T_mid, T_atm, q_mid, q_atm, gamma, **kwargs):
        """A parametric disk model as used by Williams and Best (2014)."""
        
        # User set values.
        self.r_c = r_c
        self.T_mid = T_mid
        self.T_atm = T_atm
        self.q_mid = q_mid
        self.q_atm = q_atm
        self.gamma = gamma
        
        # System related defaults.
        self.msun = 1.989e30
        self.mdisk = kwargs.get('mdisk', 0.01) * self.msun
        self.mstar = kwargs.get('mstar', 1.) * self.msun
        
        # Grid defaults.
        self.dgrid = kwargs.get('dgrid', 1.0)
        self.r_min = kwargs.get('r_min', self.dgrid)
        self.r_max = kwargs.get('r_max', 15.)
        self.z_max = kwargs.get('z_max', 10.)
        self.frame = kwargs.get('frame', 10)
        self.shape = kwargs.get('shape', 'full')
        
        # Tempearture and density related defaults.
        self.nzq = kwargs.get('nzq', 4.0)
        self.mu = kwargs.get('mu', 2.34)
        self.delta = kwargs.get('delta', 1.0)
        self.gamma = kwargs.get('gamma', 1.0)
        self.mindens = kwargs.get('mindens', 1e4)    
        self.filldens = kwargs.get('filldens', 0.0)
        self.filltemp = kwargs.get('filltemp', 0.0)
        
        # Abundance related defaults.
        self.freezeout = kwargs.get('freezeout', 20.)
        self.ndiss = kwargs.get('ndiss', 1.3e21)
        self.xgas = kwargs.get('xgas', 1e-4)
        self.xice = kwargs.get('xice', 1e-9)
        
        # Calculate the model.
        self.rgrid, self.zgrid = self.get_grids()
        self.temperature = self.get_temperature(self.rgrid, self.zgrid)
        self.density = self.get_density(self.rgrid, self.zgrid)
        self.trim_grid()
        self.temperature = np.where(self.density != self.filldens, 
                                    self.temperature,
                                    self.filltemp)
        self.frozen = self.get_frozen(self.freezeout)
        self.dissociated = self.get_dissociated(self.ndiss)
        self.abundance = self.get_abundance(self.xgas, self.xice)
        self.molecular_density = self.abundance * self.density
        
        # If appropriate, mirror the model.
        if self.shape == 'full':
            self.mirror_model()
        return
    
    def mirror_array(self, array):
        """Mirror a single array. Include r=0 column and do 
        not duplicate z=0 column."""    
        array = np.vstack([np.flipud(array), array[1:]])
        array = np.hstack([np.zeros(array.shape[0])[:, None], array])
        return np.hstack([np.fliplr(array), array[:,1:]])
    
    def mirror_model(self):
        """Mirror the model in r = 0 and z = 0 planes."""
        self.temperature = self.mirror_array(self.temperature)
        self.density = self.mirror_array(self.density)
        self.abundance = self.mirror_array(self.abundance)
        self.frozen = self.mirror_array(self.frozen)
        self.dissociated = self.mirror_array(self.dissociated)
        self.molecular_density = self.mirror_array(self.molecular_density)
        self.rgrid = np.insert(self.rgrid, 0, 0)
        self.rgrid = np.insert(self.rgrid[1:], 0, -1*self.rgrid[::-1])
        self.zgrid = np.insert(self.zgrid[1:], 0, -1*self.zgrid[::-1])
        return
    
    def trim_grid(self):
        """Trim the grids to get rid of excess edges."""
        disk = np.where(self.density != self.filldens, 1, 0)
        trim_z = np.sum(disk, 1)
        trim_r = np.sum(disk, 0)
        flag = 0 
        if trim_r[-1] != 0:
            print 'Radial grid not large enough. Increase r_max.'
            flag +=1
        if trim_z[-1] != 0:
            print 'Vertical grid not large enough. Increase z_max.'
            flag += 1
        if flag:
            return
        for z in range(1,trim_z.size+1):
            if trim_z[-z] > 0:
                break
        for r in range(1,trim_r.size+1):
            if trim_r[-r] > 0:
                break
        if (z < self.frame or r < self.frame):
            return
        z = trim_z.size - z + self.frame
        r = trim_r.size - r + self.frame
        self.zgrid = self.zgrid[:z]
        self.rgrid = self.rgrid[:r]
        self.temperature = self.temperature[:z,:r]
        self.density = self.density[:z,:r]
        return
    
    def get_grids(self):
        """Returns the r and z grids which things are calculated."""
        rgrid = np.arange(self.dgrid, self.r_c * self.r_max + self.dgrid, self.dgrid)
        zmax = self.scaleheight(rgrid[1:]).max()
        zgrid = np.arange(0, zmax * self.z_max + self.dgrid, self.dgrid)
        return rgrid, zgrid
    
    def soundspeed(self, rgrid, zgrid):
        """Soundspeed of the gas in [m/s]."""
        return np.sqrt(sc.k * self.get_temperature(rgrid, zgrid) / self.mu / sc.m_p)
    
    def scaleheight(self, rgrid):
        """Pressure scale height of gas in [au]."""
        Tmid = self.temp_midplane(rgrid)
        Hp = sc.k * Tmid * rgrid**3 * sc.au / self.mu / sc.m_p / sc.G / self.mstar
        return np.sqrt(Hp)
    
    def surfacedensity(self, r):
        """Surface density of gas [g / sqcm]."""       
        sigma = np.exp(self.powerlaw(r, self.r_c, -1., 2.-self.gamma))
        sigma *= self.powerlaw(r, self.r_c, 1., -self.gamma)
        sigma *= self.mdisk * (2. - self.gamma)
        sigma /= 2. * np.pi * np.power(self.r_c * sc.au * 100, 2.) / 1e3
        return np.where(r >= self.r_min, sigma, 0.0)
    
    def temp_midplane(self, r):
        """Returns the midplane temperature [K] at r [au]."""
        temp = self.powerlaw(r, self.r_c, self.T_mid, -self.q_mid)
        return np.where(r >= self.r_min, temp, 0.0)
    
    def temp_atmosphere(self, r):
        """Returns the atmospheric temperature [K] at r [au]."""
        temp =  self.powerlaw(r, self.r_c, self.T_atm, -self.q_atm)
        return np.where(r >= self.r_min, temp, 0.0)
    
    def get_temperature(self, rgrid, zgrid):
        """Calculate the temperature on the provided grid axes."""
        temp_mid = self.temp_midplane(rgrid)
        temp_atm = self.temp_atmosphere(rgrid)
        zq = self.nzq * self.scaleheight(rgrid)
        T = np.sin(np.pi * zgrid[:, None] / 2. / zq[None, :])
        T = np.power(T, 2. * self.delta)
        T *= temp_atm[None, :] - temp_mid[None, :]
        T += temp_mid[None, :]
        return np.where(zgrid[:, None] > zq[None, :], temp_atm[None, :], T)    
    
    def get_density(self, rgrid, zgrid):
        """Calculate the density [n(H2)] structure on the provided grid axes."""
        T = np.log(self.get_temperature(rgrid, zgrid + self.dgrid))
        T -= np.log(self.get_temperature(rgrid, zgrid - self.dgrid))
        T /= 2. * self.dgrid * sc.au
        T = np.where(np.isfinite(T), T, 0.0)
        cs = np.power(self.soundspeed(rgrid, zgrid), -2.)
        cs = np.where(np.isfinite(cs), cs, 0.0)
        grav = sc.G * self.mstar * zgrid * sc.au
        grav = grav[:, None] / np.hypot(rgrid[None, :] * sc.au, zgrid[:, None] * sc.au)**3.
        grav = np.where(np.isfinite(grav), grav, 0.0)
        # TODO: Is there a way to speed this bit up?
        rho = np.ones(T.shape)
        for i in range(0, rgrid.size):
            for j in range(1, zgrid.size):
                rho[j,i] = np.log(rho[j-1,i]) 
                rho[j,i] -= self.dgrid * sc.au * (grav[j,i] * cs[j,i] + T[j,i]) 
                rho[j,i] = np.exp(rho[j,i])
        rho -= np.nanmin(rho)
        rho /= np.trapz(rho, zgrid * sc.au * 100., axis=0)
        rho *= self.surfacedensity(rgrid)[None, :] 
        rho /= self.mu * sc.m_p * 1e3
        return np.where(rho >= self.mindens, rho, self.filldens)

    def get_abundance(self, gas_abundance, ice_abundance):
        """Return the molecular abundance."""
        return np.where(self.frozen + self.dissociated == 0, gas_abundance, ice_abundance)
    
    def get_frozen(self, freezeout_temp):
        """Return boolean of if the molecule is frozen."""
        return np.where(self.temperature <= freezeout_temp, 1, 0)
    
    def get_dissociated(self, ndiss):
        """Return a boolean of if the molecule is shielded."""
        shield = [[np.trapz(self.density[j:,i], self.zgrid[j:]*sc.au*100.)
                       for i in range(self.rgrid.size)]
                      for j in range(self.zgrid.size)]
        shield = np.array(shield)
        return np.where(shield <= ndiss, 1, 0)
    
    def interp(self, rpnts, zpnts, parameter, **kwargs):
        """Interpolates the 2D array: parameter."""
        ridx = np.interp(rpnts, self.rgrid, np.arange(self.rgrid.size))
        zidx = np.interp(zpnts, self.zgrid, np.arange(self.zgrid.size))
        return map_coordinates(parameter, [zpnts, rpnts], order=1, **kwargs)
    
    @staticmethod
    def powerlaw(x, x0, a, b):
        return a * np.power(x / x0, b)

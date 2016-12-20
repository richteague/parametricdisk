import numpy as np
import scipy.constants as sc
from scipy.ndimage.interpolation import map_coordinates

###
class ray:
    """Particular ray."""

    def __init__(self, coords, model):
        self._rpnts = coords[0]
        self._zpnts = coords[1]
        self._density = model.interp('density', self._rpnts, self._zpnts, order=1, cval=0)
        self._temperature = model.interp('temperature', self._rpnts, self._zpnts, order=1, cval=0)
        self._abundance = model.interp('abundance', self._rpnts, self._zpnts, order=0, cval=0)
        return
        
    @property
    def zpnts(self):
        return self._zpnts
        
    @property
    def rpnts(self):
        return self._rpnts
        
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def density(self):
        return self._density

    @property
    def abundance(self):
        return self._abundance
        
    @property
    def weights(self):
        n_mol = self._abundance * self._density
        if sum(n_mol) == 0:
            n_mol += np.random.rand(n_mol.size) * 1e-10
        return n_mol


###
class tracer:
    """
    Trace rays through a disk model and return physical 
    properties along those rays.
    """
    
    def __init__(self, ppd_instance):
        self.model = ppd_instance
        self._rays = {}
        return 

    def get_ray(self, intercept, angle):
        """Returns a ray instance."""
        try:
            return self._rays[intercept, angle]
        except:
            pass
        coords = self.ray_coords(intercept, angle)            
        self._rays[intercept, angle] = self.ray_trace(coords)
        return self._rays[intercept, angle]
    
    def ray_coords(self, intercept, angle):
        """Returns the (r,z) coordinates of the ray."""
        z0 = np.nanmin(self.model.zgrid)
        z1 = np.nanmax(self.model.zgrid)
        dr = np.tan(np.radians(angle)) * z1
        r0 = intercept - dr
        r1 = intercept + dr
        nsteps = np.ceil(np.hypot((r1 - r0), (z1 - z0)) / self.model.dgrid)
        rpnts = np.linspace(r0, r1, nsteps)
        zpnts = np.linspace(z0, z1, nsteps)
        in_grid = np.logical_and(rpnts >= self.model.rgrid.min(), rpnts <= self.model.rgrid.max())
        rpnts = rpnts[in_grid]
        zpnts = zpnts[in_grid]
        return rpnts, zpnts
    
    def ray_trace(self, coords):
        """Returns a dictionary along the path."""
        return ray(coords, self.model)

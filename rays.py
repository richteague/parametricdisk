import numpy as np
import scipy.constants as sc
from disk import ppd
from scipy.ndimage.interpolation import map_coordinates

class rays:
    """Trace rays through a disk model and return physical 
    properties along those rays."""
    
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
        ray = {}
        ray['rpnts'] = coords[0]
        ray['zpnts'] = coords[1]
        ray['density'] = self.model.interp(coords[0], coords[1], self.model.density)
        ray['temperature'] = self.model.interp(coords[0], coords[1], self.model.temperature)
        ray['abundance'] = self.model.interp(coords[0], coords[1], self.model.abundance)
        return ray

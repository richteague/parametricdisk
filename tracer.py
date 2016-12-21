import numpy as np


class ray:
    """Ray through the disk model."""

    def __init__(self, coords, model, **kwargs):

        # Coordinates of the path through the disk.
        self._rpnts = coords[0]
        self._zpnts = coords[1]

        # Three main physical properties of the disk.
        self._density = model.interp('density',
                                     self._rpnts,
                                     self._zpnts,
                                     order=1,
                                     cval=0)

        self._temperature = model.interp('temperature',
                                         self._rpnts,
                                         self._zpnts,
                                         order=1,
                                         cval=0)

        self._abundance = model.interp('abundance',
                                       self._rpnts,
                                       self._zpnts,
                                       order=0,
                                       cval=0)

        # Abundance weights.
        # Include some random values so sum(weights) != 0.
        self._weights = self._abundance * self._density
        if sum(self._weights) == 0:
            self._weights += np.random.rand(self._weights.size) * 1e-30
        return

    # Easy to access properties of the ray.
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
        return self._weights

    @property
    def temp_avg(self):
        return np.average(self.temperature, weights=self.weights)

    @property
    def dens_avg(self):
        return np.average(self.density, weights=self.weights)

    @property
    def temp_std(self):
        to_avg = (self.temp_avg - self.temperature)**2.
        return np.average(to_avg, weights=self.weights)**0.5

    @property
    def dens_std(self):
        to_avg = (self.dens_avg - self.density)**2.
        return np.average(to_avg, weights=self.weights)**0.5


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
        rmin = self.model.rgrid.min()
        rmax = self.model.rgrid.max()
        in_grid = np.logical_and(rpnts >= rmin, rpnts <= rmax)
        rpnts = rpnts[in_grid]
        zpnts = zpnts[in_grid]
        return rpnts, zpnts

    def ray_trace(self, coords):
        """Returns a dictionary along the path."""
        return ray(coords, self.model)

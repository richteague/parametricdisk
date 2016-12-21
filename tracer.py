import numpy as np


class ray:
    """Ray through the disk model."""

    def __init__(self, coords, model, **kwargs):

        # Coordinates of the path through the disk.
        self.rpnts = coords[0]
        self.zpnts = coords[1]

        # Three main physical properties of the disk.
        self.density = model.interp('density',
                                    self.rpnts,
                                    self.zpnts,
                                    order=1,
                                    cval=0)

        self.temperature = model.interp('temperature',
                                        self.rpnts,
                                        self.zpnts,
                                        order=1,
                                        cval=0)

        self.abundance = model.interp('abundance',
                                      self.rpnts,
                                      self.zpnts,
                                      order=1,
                                      cval=0)

        # Abundance weights.
        # Include some random values so sum(weights) != 0.
        self.weights = self.abundance * self.density
        if sum(self.weights) == 0:
            self.weights += np.random.rand(self.weights.size) * 1e-30
        return

    # Easy to access properties of the ray.
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

    def statistics(self, param, percentiles=[0.16, 0.5, 0.84]):
        """Returns the weighted percentiles of the param value."""

        # Select the appropriate physical parameter.
        if param in 'density':
            arr = self.density.copy()
        elif param in 'temperature':
            arr = self.temperature.copy()

        # Apply the weighted percentiles.
        # Sort the data.
        i = np.argsort(arr)
        n = arr.shape[0]

        sd = np.take(arr, i)
        sw = np.take(self.weights, i)
        aw = np.add.accumulate(sw)
        w = (aw - 0.5 * sw) / aw[-1]

        spots = np.searchsorted(w, percentiles)
        vals = []
        for (s, p) in zip(spots, percentiles):
            if s == 0:
                vals.append(sd[0])
            elif s == n:
                vals.append(sd[n-1])
            else:
                f1 = (w[s] - p) / (w[s] - w[s-1])
                f2 = (p - w[s-1]) / (w[s] - w[s-1])
                vals.append(sd[s-1] * f1 + sd[s] * f2)

        return np.array(vals)


class tracer:
    """
    Trace rays through a disk model and return physical
    properties along those rays.
    """

    def __init__(self, ppd_instance):
        """Initialise with a model instance."""
        self.model = ppd_instance
        self._rays = {}
        return

    def get_ray(self, intercept, angle, ds=1.0):
        """Returns a ray instance. Intercept in [au], angle in [deg]."""
        try:
            return self._rays[intercept, angle, ds]
        except:
            pass
        coords = self.ray_coords(intercept, angle, ds)
        self._rays[intercept, angle, ds] = self.ray_trace(coords)
        return self._rays[intercept, angle, ds]

    def ray_coords(self, intercept, angle, ds):
        """Returns the (r,z) coordinates of the ray in [au]."""
        z0 = np.nanmin(self.model.zgrid)
        z1 = np.nanmax(self.model.zgrid)
        dr = np.tan(np.radians(angle)) * z1
        r0 = intercept - dr
        r1 = intercept + dr
        nsteps = np.ceil(np.hypot((r1 - r0), (z1 - z0)) / ds)
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

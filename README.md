# parametricdisk
Quickly build and analyse simple parametric models of protoplanetary disks.

##disk.py
Generates parametirc models of protoplanetary disks following the prescription in [Williams and Best (2014)](https://ui.adsabs.harvard.edu/#abs/2014ApJ...788...59W). All parameters are changeable however a basic model can be generated through:

```python
model = pdd(r_c=30., t_mid=20., t_atm=150., q_mid=0.5, q_atm=0.5, gamma=1.0)
```

Properties can then be easily accessed and plotted:

```python
fig, ax = plt.subplots()
im = ax.contourf(model.rgrid, model.zgrid, model.temperature)
plt.colorbar(im)
```

##rays.py
Quickly explore the physical properties along a ray through the disk model.

```python
rays = rays(model)
ray = rays.get_ray(intercept_radius=50., incercept_angle=30.)
```
Note that the `intercept_radius` is in au and `intercept_angle` is in degrees, both of which can be positive and negative. Again the properties are easily accessible.

```python
fig, ax = plt.subplots()
ax.plot(ray['rpnts'], ray['temperature']
```

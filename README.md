# iht

## Usage
### Setup
Ensure that the required [Dependencies](#Dependencies) are installed.

To get started, clone the repository with 

    git clone https://github.com/isulta/iht

Add `iht` to your Python path with

     export PYTHONPATH="${PYTHONPATH}:<PATH TO IHT>"

replacing \<PATH TO IHT\> with the correct path.

### Halo Centering

    from iht.iht import halo_center_wrapper
    posC = halo_center_wrapper(pdata)

`pdata` should be a dictionary that contains the particle data (coordinates and masses), where `pdata['Coordinates']` is an Nx3 array of the (x,y,z) coordinates of each of the N particles, and `pdata['Masses']` is a 1D array of length N with the particle masses.

*In my analysis, I pass in the high-resolution dark matter particles (particle type 1) for `pdata`, and centering based on just those particles works fine.*

`posC` will be a 1D array of length 3 containing the (x,y,z) coordinates of the halo center. The halo center returned is in the same units as the `pdata['Coordinates']` array.

### Virial Radius and Virial Mass

    from iht.iht import find_Rvir_SO
    Rvir, Mvir = find_Rvir_SO(part, posC)

`part` is a *dictionary of dictionaries* containing all of the particle data at a snapshot, i.e. the keys of `part` are integers $\in \{ 0,1,2,4,5 \}$ specifying the particle type (see FIRE Wiki); the values are dictionaries of the form `{'Coordinates':Nx3 array, 'Masses': 1D array of length N}`. 
The units of `'Coordinates'` should be in physical kpc, and `'Masses'` should be in $10^{10} M_\odot$.

The `part[0]` dictionary should also contain `part[0]['Omega_Matter']`, `part[0]['Omega_Lambda']`, `part[0]['HubbleParam']`, and `part[0]['Redshift']` (note that the first two of these parameters are named differently in FIRE-2 simulations).

`posC` is a 1D array of length 3 containing the (x,y,z) coordinates of the halo center, in units of physical kpc.

`Rvir` (virial radius calculated using spherical overdensity definition) will be returned in units of physical kpc; the virial mass `Mvir` (mass within a sphere of radius `Rvir` centered at `posC`) is returned in units $M_\odot$.

### Halo Potential and Circular Velocity

    from iht.potential import Potential_FIRE
    from astropy import units
    
    potential = Potential_FIRE(part) # initialize a Potential_FIRE instance for a FIRE snapshot
    vc_vir = potential.vc(potential.Rvir) # circular velocity at Rvirial
    vc_200kpc = potential.vc(200*units.kpc) # circular velocity at 200 physical kpc

`part` is a *dictionary of dictionaries* containing all of the particle data at a snapshot, i.e. the keys of `part` are integers $\in \{ 0,1,2,4,5 \}$ specifying the particle type (see FIRE Wiki); the values are dictionaries of the form `{'Coordinates':Nx3 array, 'Masses': 1D array of length N}`. 
The units of `'Coordinates'` should be in physical kpc, and `'Masses'` should be in $10^{10} M_\odot$.
`'Coordinates'` must be the *centered coordinates* relative to the halo center.

Once a `Potential_FIRE` instance is initialized, the potential $\Phi(r)$, circular velocity $v_c(r)$, and slope of the $v_c$ profile, $\mathrm{d}(\ln v_c)/\mathrm{d}(\ln r)$, can be calculated at a given radius $r$ using the `potential.Phi`, `potential.vc`, and `potential.dlnvc_dlnR` methods.
The methods accept one argument `r`, which must be a scalar or array *with units*, e.g. `200*units.kpc` or `np.arange(100,1000,100)*units.kpc`.

There are also functions to calculate properties of circular orbits:
* `potential.Ecirc(r)` returns the specific energy of a circular orbit at radius $r$
* `potential.rcirc(E)` returns the radius of a circular orbit with specific energy $E$

### Radiative cooling luminosity

    from iht.iht import luminosities
    Luminosity = luminosities(part)

Calculates radiative cooling luminosities of gas particles, in units erg/s.
See the documentation for this function in `iht.py`.
`part` is a dictionary of gas particles.

### Hot phase particle selection

    from iht.iht import find_virial_branch_particles
    hot_particle_indices = find_virial_branch_particles(r_scaled, Temperature, nH, Masses)

This function finds all gas particles belonging to the hot, virial phase in the halo, and returns their indices.
See the documentation for this function in `iht.py`.

## Dependencies
Python 3 is required. Running in a [conda](https://conda.io/projects/conda/en/latest/index.html) environment is recommended.

The following packages are also required:
- NumPy
- Numba
- Matplotlib
- SciPy
- Astropy
- [cooling flow package by Stern et al.](https://sites.northwestern.edu/jonathanstern/the-cooling_flow-package/)
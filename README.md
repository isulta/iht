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

## Dependencies
Python 3 is required. Running in a [conda](https://conda.io/projects/conda/en/latest/index.html) environment is recommended.

The following packages are also required:
- NumPy
- Numba
- Matplotlib
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

    from iht import halo_center_wrapper
    posC = halo_center_wrapper(pdata)

`pdata` should be a dictionary that contains the particle data (coordinates and masses), where `pdata['Coordinates']` is an Nx3 array of the (x,y,z) coordinates of each of the N particles, and `pdata['Masses']` is a 1D array of length N with the particle masses.

*In my analysis, I pass in the high-resolution dark matter particles (particle type 1) for `pdata`, and centering based on just those particles works fine.*

`posC` will be a 1D array of length 3 containing the (x,y,z) coordinates of the halo center. The halo center returned is in the same units as the `pdata['Coordinates']` array.

## Dependencies
Python 3 is required. Running in a [conda](https://conda.io/projects/conda/en/latest/index.html) environment is recommended.

The following packages are also required:
- NumPy
- Numba
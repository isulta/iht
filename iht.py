import numpy as np
import matplotlib.pyplot as plt
from numba import njit

### Halo centering ###
def center_of_mass(coords, masses):
    return np.array([np.sum((coords[:,i] * masses)) for i in range(3)])/np.sum(masses)

@njit
def dist(r1, r0):
    res = np.zeros_like(r1[:,0])

    rrel = r1 - r0
    for i in range(len(res)):
        for j in range(3):
            res[i] += rrel[i][j]**2
        res[i] = np.sqrt(res[i])
    return res

def halo_center(coords, masses, shrinkpercent=2.5, minparticles=1000, initialradiusfactor=1, verbose=False):
    '''See Power et al. 2003 for shrinking sphere halo centering algorithm details.

    most accurate parameters: shrinkpercent=2.5, minparticles=1000, initialradiusfactor=1
    These are the default and recommended parameters, and also the parameter values of Power et al. 2003.

    pretty accurate parameters and 2x faster than shrinkpercent=2.5 (some minor problems, e.g. a small dip at z=6 for h29_noAGNfb): 
    shrinkpercent=10, minparticles=1000, initialradiusfactor=1

    fastest parameters; fail to find correct center in some cases:
    shrinkpercent=50, minparticles=1000, initialradiusfactor=0.25
    '''
    com = center_of_mass(coords, masses)

    r = dist(coords, com)
    
    radius = r.max()*initialradiusfactor

    Nconverge = min(minparticles, len(masses)*0.01)
    iteration = 0

    coords_it = coords.copy()
    masses_it = masses.copy()

    comlist = [com]
    radiuslist = [radius]

    while len(masses_it) > Nconverge:
        radius *= (100-shrinkpercent)/100

        mask = r <= radius
        coords_it = coords_it[mask, :]
        masses_it = masses_it[mask]

        com = center_of_mass(coords_it, masses_it)

        r = dist(coords_it, com)
        
        iteration += 1
        comlist.append(com)
        radiuslist.append(radius)

        if verbose:
            print(iteration, radius, np.format_float_scientific(len(masses_it)), com)
    
    return com, comlist, radiuslist

def halo_center_wrapper(pdata, shrinkpercent=2.5, minparticles=1000, initialradiusfactor=1):
    coords = pdata['Coordinates']
    masses = pdata['Masses']
    return halo_center(coords, masses, shrinkpercent, minparticles, initialradiusfactor)[0]

'''Finds virial radius (units physical kpc) given particle dict and halo center, using spherical overdensity definition.
If `halo` and `snapnum` are defined, a plot of density vs. distance from halo center is saved.
The virial mass (mass within a sphere of radius Rvir centered at posC) is also returned in units Msun.
'''
def find_Rvir_SO(part, posC=None, halo=None, snapnum=None):
    if posC is None:
        posC = part[0]['posC']

    Masses = []
    r = []
    for ptype, p_i in part.items():
        if not 'Masses' in p_i.keys(): #some simulations don't have ptype 5 (black holes)
            print(f'{halo}: Masses not found for ptype {ptype} at snapshot {snapnum}')
            continue
        
        Masses.append(p_i['Masses'])
        
        # position relative to center
        p_i_CoordinatesRelative = p_i['Coordinates'] - posC

        # distance from halo center
        p_i_r = np.linalg.norm(p_i_CoordinatesRelative, axis=1)

        r.append(p_i_r)
    
    Masses = np.concatenate(Masses)
    r = np.concatenate(r)

    idx = np.argsort(r)
    Masses = Masses[idx]
    r = r[idx]
    Volume = 4/3 * np.pi * r**3 # Volume in units (physical kpc)^3

    Masses = np.cumsum(Masses) * 1.e10 # Total mass in units Msun within sphere of radius r

    with np.errstate(divide='ignore'): Density = Masses/Volume * 1.e9 # Density in units Msun/Mpc^3

    if 'Omega0' in part[0]: #FIRE-2
        OmegaM0, OmegaL0, hubble, z = part[0]['Omega0'], part[0]['OmegaLambda'], part[0]['HubbleParam'], part[0]['Redshift']
    else: #FIRE-3
        OmegaM0, OmegaL0, hubble, z = part[0]['Omega_Matter'], part[0]['Omega_Lambda'], part[0]['HubbleParam'], part[0]['Redshift']
    rhovir = deltavir(OmegaM0, OmegaL0, z) * rhocritz(OmegaM0, OmegaL0, z) * hubble**2 # Virial density in units Msun/Mpc^3

    if halo is not None:
        plt.plot(r, Density)
        plt.yscale('log')
        plt.axhline(rhovir, label=r'$\Delta_{vir} \rho_{crit}$')
        plt.xlim(0,300)
        plt.xlabel('r (pkpc)')
        plt.ylabel('Density (Msun/pMpc^3)')
        plt.legend()
        plt.savefig(f'density_{halo}_snapnum_{snapnum}.png')
        plt.close()
    
    idx_vir = np.flatnonzero(Density <= rhovir)[0]
    return r[idx_vir], Masses[idx_vir] # return Rvir in units physical kpc, and Mvir in units Msun
    # simple linear interpolation with next closest point, and InterpolatedUnivariateSpline.roots() both seem to return approximately same Rvir as the 1 point method above.

### COSMOLOGY CODE ###
def scale_factor_to_redshift(a):
    z = 1/a - 1
    return z

def Ez(OmegaM0, OmegaL0, z):
    return ( OmegaM0*(1+z)**3 + OmegaL0 )**0.5

def OmegaM(OmegaM0, OmegaL0, z):
    return OmegaM0 * (1+z)**3 / Ez(OmegaM0, OmegaL0, z)**2

def xz(OmegaM0, OmegaL0, z):
    return OmegaM(OmegaM0, OmegaL0, z) - 1

def deltavir(OmegaM0, OmegaL0, z):
    '''Virial overdensity fitting function from Bryan & Norman (1998)'''
    x = xz(OmegaM0, OmegaL0, z)
    return 18*np.pi**2 + 82*x - 39*x**2

def rhocritz(OmegaM0, OmegaL0, z):
    '''Returns the critical density at z in units h^2 Msun/Mpc^3.'''
    rhocrit0 = 2.77536627e11 #h^2 Msun/Mpc^3
    return rhocrit0 * Ez(OmegaM0, OmegaL0, z)**2
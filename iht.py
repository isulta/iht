import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.spatial.transform import Rotation as R
from astropy import units as un, constants as cons

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

    Masses = np.cumsum(Masses, dtype=np.float64) * 1.e10 # Total mass in units Msun within sphere of radius r

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

'''Finds velocity of the halo's center of mass, defined as mass-weighted average velocity of ALL particles within r < Rmax*Rvir.
'''
def velocity_COM(part, Rmax=1):
    # Get COM's (r/Rvir<Rmax) velocity 
    pall = {}
    pall['Masses'] = np.concatenate([part[k]['Masses'] for k in part.keys()])
    pall['Velocities'] = np.concatenate([part[k]['Velocities'] for k in part.keys()])
    pall['r_scaled'] = np.concatenate([part[k]['r_scaled'] for k in part.keys()])

    maskall = pall['r_scaled']<Rmax
    velC = np.sum( (pall['Velocities'].T * pall['Masses']).T[maskall], axis=0 ) / np.sum( pall['Masses'][maskall] )
    return velC

def inrange(a, ra, right_bound_inclusive=True):
    a1, a2 = ra
    if right_bound_inclusive:
        return (a1 <= a)&(a <= a2)
    else:
        return (a1 <= a)&(a < a2)

def continuous_mode(data, bins=100, range=None):
    # Create a histogram
    hist, bin_edges = np.histogram(data, bins=bins, range=range)
    
    # Find the bin with the highest frequency (mode)
    mode_index = np.argmax(hist)
    
    # The mode value will be the midpoint of the mode bin
    mode = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
    
    return mode

def find_virial_branch_particles( r_scaled, Temperature, nH, Masses, rbins=np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(3), 0.05)), mainBranchHalfWidth=0.5, fbranchcut=0.5 ):
    '''Selects all gas particles in the main, virial branch of the halo. Returns a 1D array of particle indices.

    `r_scaled` is 1D array of the radial coordinate of each particle (distance relative to halo center), in units of the virial radius.
    `Temperature` is 1D array of particle temperatures, in units K.
    `nH` is 1D array of particle hydrogen number densities, in units cm^-3.
    `Masses` is 1D array of particle masses.

    `mainBranchHalfWidth` is half of the width (in log space) of the virial branch.
    
    Radial shells for which the mass fraction of particles in the virial branch is less than `fbranchcut` are excluded.
    
    Default `rbins` chosen to match Stern+20 Fig. 6: `np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(1.9597976388995666), 0.05))`
    '''
    rmid = (rbins[:-1]+rbins[1:])/2 #midpoint of radial shells, in units of Rvir
    Tmask = (Temperature > 10**5.0) #exclude particles with T <= 1e5 K from mode calculation

    idx_virial_allbins = []

    for r0,r1 in zip(rbins[:-1],rbins[1:]):
        rmask = inrange( r_scaled, (r0, r1) )
        Mshell = np.sum(Masses[rmask]) #mass of all particles in shell

        idx = np.flatnonzero(Tmask & rmask)
        T_mode = continuous_mode(np.log10(Temperature[idx]), range=(3,8)) #log space
        nH_mode = continuous_mode(np.log10(nH[idx]), range=(-7,0)) #log space

        idx_virial = np.flatnonzero(rmask&
                                inrange( np.log10(Temperature), (T_mode - mainBranchHalfWidth, T_mode + mainBranchHalfWidth) )&
                                inrange( np.log10(nH), (nH_mode - mainBranchHalfWidth, nH_mode + mainBranchHalfWidth) ))
        
        Mshell_virial = np.sum(Masses[idx_virial]) #mass of particles in shell belonging to virial branch
        fbranch = Mshell_virial / Mshell #mass fraction of virial branch relative to all particles in shell

        if fbranch >= fbranchcut:
            idx_virial_allbins.append(idx_virial)
    
    return np.concatenate(idx_virial_allbins)

def find_angmom_vector(pos, vel, mass, normalize=True):
    angmom = np.sum( np.cross(pos, vel) * mass[:, None], axis=0 )
    if normalize: angmom /= np.linalg.norm(angmom)
    return angmom

def rotate_coords(pdata, Rvirial, Rgal=0.1):
    '''Given dictionary of particles, rotates coordinates and velocities such that the new z-axis is the axis of rotation of the galaxy.
    `pdata[k]['Coordinates']` and `pdata[k]['Velocities']` are changed to be the new rotated coordinates.
    
    `pdata[k]['Coordinates']` and `pdata[k]['Velocities']` must be defined relative to the center of halo before calling this function.
    
    The axis of rotation of the galaxy is calculated as the total angular momentum of all star particles (`pdata[4]`) within `Rgal`*`Rvirial`.
    `Rvirial` must have the same units as `pdata[k]['Coordinates']`.
    '''
    # Net angular momentum vector of all star particles within Rgal*Rvirial
    idx = np.linalg.norm(pdata[4]['Coordinates'], axis=1) < (Rgal*Rvirial)
    j = find_angmom_vector(pdata[4]['Coordinates'][idx], pdata[4]['Velocities'][idx], pdata[4]['Masses'][idx])
    
    u = np.cross(j, [0,0,1])
    if np.linalg.norm(u) == 0: return # j is already aligned with z-axis
    
    u /= np.linalg.norm(u)
    theta = np.arccos(j[2] / np.linalg.norm(j))
    
    r = R.from_rotvec(theta * u)
    
    for k in pdata.keys():
        pdata[k]['Coordinates'] = r.apply(pdata[k]['Coordinates'])
        pdata[k]['Velocities'] = r.apply(pdata[k]['Velocities'])

def spherical_velocities(v, r):
    '''Given velocity v and position r cartesian arrays, calculates vrad, vtheta, vphi (each in same units as v).
    For N particles, v is Nx3 array of cartestian particle velocities, and r is a Nx3 array of carestian particle positions.
    `v` and `r` must be defined relative to the center of halo before calling this function.
    '''
    theta = np.arctan2( np.sqrt(r[:,0]**2 + r[:,1]**2), r[:,2] )
    phi = np.arctan2( r[:,1], r[:,0] )
    rhat = np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
    thetahat = np.column_stack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)))
    phihat = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    
    vrad = np.sum(v*rhat, axis=1)
    vtheta = np.sum(v*thetahat, axis=1)
    vphi = np.sum(v*phihat, axis=1)
    
    return vrad, vtheta, vphi

def luminosities(part, Zbins=1000):
    """Calculates particle luminosities, in units erg/s.

    `part` is a dictionary of gas particles:
        `part['nH']` is hydrogen number density in units cm^-3.
        `part['Temperature']` is temperature in units K.
        `part['Metallicity']` is the mass fraction of metals; this is the prop('massfraction.metals') field in gizmo_analysis.
        `part['Volume']` is particle volume in units physical kpc^3.
        `part['Redshift']` is the redshift of the snapshot.

    This function requires the cooling flow package by Stern et al. https://sites.northwestern.edu/jonathanstern/the-cooling_flow-package/
    Make sure to set the `dataDir` variable in `WiersmaCooling.py` to the correct path of the cooling/ subfolder, 
    e.g. `dataDir = '/home/ias627/tools/cooling_flow/cooling/'`

    Cooling rates are calculated using the Wiersma et al. 2009 cooling tables (with Z=mean Z in 1000 metallicity bins).
    """

    import WiersmaCooling as Cool
    Zsun = 0.0142 #Asplund+09
    Zidxsplit = np.array_split( np.argsort(part['Metallicity']/Zsun), Zbins )
    CoolingRate = np.zeros_like(part['nH']) #cooling rate in erg/s/cm^3

    for Zidx in Zidxsplit:
        Z2Zsun = np.mean(part['Metallicity'][Zidx]) / Zsun
        cooling = Cool.Wiersma_Cooling(Z2Zsun, part['Redshift'])
        CoolingRate_pred = cooling.LAMBDA(part['Temperature'][Zidx]*un.K, part['nH'][Zidx]*(un.cm**-3)) * (part['nH'][Zidx]*(un.cm**-3))**2
        CoolingRate[Zidx] = CoolingRate_pred.to(un.erg/un.s*un.cm**-3).value

    Luminosity = CoolingRate * part['Volume']
    Luminosity = (Luminosity * un.erg/un.s*un.cm**-3 * un.kpc**3).to(un.erg/un.s).value
    return Luminosity

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
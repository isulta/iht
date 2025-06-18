import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from astropy import units as un, constants as cons

class Potential_FIRE():
    def __init__(self, part, Rmax=3):
        Mr = calculateMr(part)
        self.Rvir = Mr['Rvir'][()]*un.kpc
        
        Phir = cumtrapz( Mr['M(r)'] / Mr['r']**2,  Mr['r'], initial=0)
        Rvir3_idx = np.argmin(np.abs(Mr['r'] - Rmax*Mr['Rvir'][()]))
        Phir = Phir - Phir[Rvir3_idx]
        self.Phi_interp = interp1d(Mr['r'], Phir)
        
        vcr = np.sqrt(Mr['M(r)']/Mr['r'])
        vcr = savgol_filter(vcr, 11, 2) #changed from 10 to 11
        self.vc_interp = interp1d(Mr['r'], vcr, fill_value='extrapolate')
        
        lnvcr = np.gradient(np.log(vcr), np.log(Mr['r']))
        self.lnvc_interp = interp1d(Mr['r'], lnvcr)

    def vc(self, r):
        r = r.to(un.kpc).value
        return ( self.vc_interp(r) * (cons.G*un.Msun/un.kpc)**0.5 ).to(un.km/un.s)
    def Phi(self, r):
        r = r.to(un.kpc).value
        return self.Phi_interp(r) * (cons.G*un.Msun/un.kpc).to((un.km/un.s)**2)
    def dlnvc_dlnR(self, r):
        r = r.to(un.kpc).value
        return self.lnvc_interp(r)

def calculateMr(part, Rmin=None, Rmax=None, bins=100):
    '''Calculate the cumulative mass profile M(<r) for a given set of particles.'''
    if Rmin is None:
        Rmin = 0.01 #pkpc
        Rmax = 10 * part[1]['Rvir'] #pkpc
    rbins = np.logspace(np.log10(Rmin), np.log10(Rmax), bins) #pkpc
    # M(r)
    pall = {}
    pall['Coordinates'] = np.concatenate([part[k]['Coordinates'] for k in part.keys()]) #Assume Coordinates already centered
    pall['Masses'] = np.concatenate([part[k]['Masses'] for k in part.keys()])
    pall['r'] = np.linalg.norm(pall['Coordinates'], axis=1)

    Mrbins = np.array([np.sum(pall['Masses'][pall['r']<=r]) for r in rbins]) * 1e10 #M(r) for r in rbins where [M]=Msun and [r]=pkpc
    Mr = {'r':rbins, 'M(r)':Mrbins, 'Rvir':part[1]['Rvir']}
    return Mr
#%%
import numpy as np
from scipy.interpolate import interp1d

from matlabtools import Struct
from fullwave2d import definitions

from francesco_feature.turbulence_map import TURBULENCE, show_input_turbulence_spec

def interpolator(X, Y, nmesh):
    X_new  = np.linspace(X.min(), X.max(), nmesh)
    interp = interp1d(X, Y, kind = 'linear', fill_value = 'extrapolate')
    return X_new, interp(X_new)


def get_density(density_map, dx, nx, ny, **kwargs):
    
    ne = np.zeros([ny, nx])

    if density_map == 'linear_profile':
        #linear density profile with the cutoff at pos%
        pos_cutoff = kwargs.pop('pos_cutoff', 0.07)
        n_cutoff   = kwargs.pop('n_cutoff', 4e19)
        
        for i in range(0, nx):
            ne[:, i] = n_cutoff * i * dx / pos_cutoff
        
        return np.flip(ne, axis = 1).astype(np.double)
    
    elif density_map == 'real_profile':
        
        p = (definitions.DENSITY_DIR / str(kwargs.pop('shot', 57558))).with_suffix('.mat')
        density = Struct.from_mat(p)
        
        R_new, ne_new = interpolator(density.R, density.ne_int, nx)
        
        for i in range(0, nx):
            ne[i, :] = ne_new
        
        return np.flip(ne, axis = 0).astype(np.double)

def get_spectrum(subdir, **kwargs):
    
    if subdir == 'linear_profile':
        spectrum = kwargs.pop('spectrum', None)
        
        if spectrum == 0:
            alpha = np.array([0, 0, 0])
            k_kne = np.array([100, 400, 3000])
        elif spectrum == 1:
            alpha = np.array([0, -2, -2])
            k_kne = np.array([100, 400, 3000])
        elif spectrum == 2:
            alpha = np.array([0, -1, -4])
            k_kne = np.array([100, 500, 3000])
        
        return alpha, k_kne

def get_perturbation(subdir, dx, nx, ny_tot, **kwargs):
    
    if subdir == 'linear_profile':
        alpha, knee  = get_spectrum(subdir, **kwargs)
        turb         = TURBULENCE(dx, nx, ny_tot) 
        dn, kspec, _ = turb.from_spect(alpha, knee)

        return dn.T.astype(np.double)
    
    elif subdir == 'tilt_angle':
        turb         = TURBULENCE(dx, nx, ny_tot)
        lmin, lmax, beta = kwargs.get('lmin', 0), kwargs.get('lmax', 0), kwargs.get('beta', 0)
        dn, kspec, _ = turb.from_gaussian(lmin, lmax, beta)
        
        return dn.T.astype(np.double)

# %%

# %%

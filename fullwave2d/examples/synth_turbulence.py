#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import getpass

from pathlib import Path
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import time

from fullwave2d.core.wrapper import fw2d_wrapper, InputData
from fullwave2d.main.turbulence_map import SynthTurbMap, lin_prof, rms


    
# global simulation parameters
f0 = 60e9 # probing frequency [Hz]
nt = 6000 # number of time steps
ny,nx = data.shape
dx = 1.5e-4 # spatial resolution [m]
waist = 30 * dx
yante = (float(ny)/2-200) * dx
angle = 20.
rms = 0.02 # intensity of fluctuations as a fraction of the (local) background density



def _show_input_turbulence(data):
    
    from scipy.signal import welch
    
    
    # first plot: PSD of the input turbulence along the x and y axes
    fig, ax = plt.subplots()
    for axis, label, k in zip([1,0], ['y', 'x'], [ky, kx]):
        f, _Pxx = welch(data, fs=1/dx, axis=axis, nperseg=256, return_onesided=False, detrend=False)
        Pxx = np.mean(_Pxx, axis=int(not axis))
        
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)
        l, = ax.plot(f / 100, Pxx / Pxx.max(), label=label)
        ax.axvline(k / 100, ls='--', c=l.get_color())
    ax.axhline(np.exp(-1), ls='--', c='k')
    ax.legend()
    
    # second plot: input turbulence in real space
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='seismic', interpolation='none', origin='lower')
    


def make_input(turbulence, simname):
    """
    Create a linear density profile and add turbulence on top.
    """


    # define a place holder for the density map:
    ne = np.zeros(nx * ny).reshape((ny, nx)).astype(np.double)
    # NOTE: The axes (x,y) of ne are reversed, because the source script
    # maxwell_2d_omode.c expects a 2d array of shape (ny, nx)
    
    # intitalize with linear background profile and
    # cutoff at halfway in the plasma
    lin_prof(ne, f0, cut=.7, start=0)
    ne = np.flip(ne, axis=1)
    
    dne = rms * ne * turbulence # add turbulence
    
    ne = ne + dne


    # input to the full wave script
    inp = InputData(
                    name = simname,
                    f0 = f0,
                    nt = nt,
                    nx = nx,
                    ny = ny,
                    dx = dx,
                    ne = ne,
                    waist = waist,
                    yante = yante,
                    angle = angle,
                    save_diag = True
                    )
    
    return inp

    # launch full wave iteration
    t0 = time.time()
    fw2d_wrapper(inp)
    print('time (s) : ', time.time() - t0)

    # NOTE: DO NOT TRY TO PLOT BEFORE CALLING
    # THE MAXWELL ROUTINE
    # For a strange reason, this will replace
    # decimal separators by commas in the output
    # .dat/.txt files, leading to an error when
    # trying to import it

    
#%%

# Set up homogeneous 2d Gaussian turbulence map (slightly anisotropic)

ky = 8.6e2 # [rad/m]
kx = 10.e2 # [rad/m]

ly = 1./ky # m
lx = 1./kx # m
beta = 0 * pi / 180 # rad 0 means no inclination of eddies

map_args = (dx, nx, ny, lx, ly, beta)
map = SynthTurbMap(map_args)
Gauss_turbulence = map.delta_ne
    
#%%
if __name__ == '__main__':
    pass
    
    # display input turbulence 
    _show_input_turbulence(Gauss_turbulence)
    

    #%% computation
    for data, simname in zip([Gauss_turbulence], ['Gauss_turbulence_example']):
        plt.close('all') # to avoid a strange error when running the code
        inp = make_input(data, simname)
        # launch full wave iteration
        t0 = time.time()
        fw2d_wrapper(inp)
        print('time (s) : ', time.time() - t0)

        
    #%% display results
    for simname in ['Gauss_turbulence_example']:
        InputData.display_results([simname]);

# %%

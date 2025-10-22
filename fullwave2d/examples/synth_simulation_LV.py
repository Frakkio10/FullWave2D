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

# select data path depending on username:
user = getpass.getuser()
if user=='sascha': # assumes local PC
    p = Path.home() / 'data/lvermare/flat_MAP1_r.h5'
else: # assumes path on Marconi
    p = '/home/LV219680/flat_MAP1_r.h5'

with h5py.File(p, 'r') as f:
    data = np.array(f['/fields/fluct'])
    # normalize the turbulence (if not already done):
    #data = ( data - np.mean(data) ) / np.std(data)

# rename 
flat_turbulence = data
    
# global simulation parameters
f0 = 60e9 # probing frequency [Hz]
nt = 6000 # number of time steps
ny,nx = data.shape
dx = 1.5e-4 # spatial resolution [m]
waist = 30 * dx
yante = (float(ny)/2-200) * dx
angle = 20.
rms = 0 # intensity of fluctuations as a fraction of the (local) background density


#%%

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
    #lin_prof(ne, f0, cut=.7, start=0)
    nfact=4e19
    for i in range(nx):
            ne[:,i] = nfact*i*dx /0.07
    
    ne = np.flip(ne, axis=1)
    
    dne = rms * nfact * turbulence # add turbulence
    
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

 

inp = make_input(data,'test_LV_woturb')
        # launch full wave iteration
t0 = time.time()
fw2d_wrapper(inp)
print('time (s) : ', time.time() - t0)   
#%%


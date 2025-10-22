#%%
from fullwave2d.main.mpi_maxwell import scatterv_maxwell_from_h5
from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData
# from fullwave2d import definitions
import time
from mpi4py import MPI
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as C
from matplotlib import colors
import pickle
import h5py as h5

#%% 

# Set up the parallelization 

simulations_per_CPU = 1
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root
save_diag = True if is_root else False

# %%

filename = '/home/forlacchio/FlowWaveSim/output/h5/tilt/advection_only.h5'
with h5.File(filename, "r") as f:
    Nt, Ny, Nx = f["fields/n"].shape
    delta_ne   = f['fields/n'][()]
    x, y = f['grid/x'][()], f['grid/y'][()]
#%%
#define the inputs parameters

name            = f'advection_only'
subdir          = 'test_FWS_2'
mode            = 'O'
f0, nx, ny, dx  = 60e9, Nx, Ny, x[1]
b0              = 2.5
waist, yante    = 30 * dx, int(ny / 2 - 250) * dx
angle           = 10
spectrum        = 1
nt              = 300
B0              = (b0 * np.ones([nx, ny])).astype(np.double)

ne_lin = np.zeros([ny, nx])
#linear density profile with the cutoff at pos%
pos_cutoff = 0.08
n_cutoff   = 4e19

for i in range(0, nx):
    ne_lin[:, i] = n_cutoff * i * dx / pos_cutoff

ne_lin = np.flip(ne_lin, axis = 1)

if size == 1:
    plt.pcolormesh(x, y, ne_lin, cmap = 'terrain')
    plt.colorbar()
# %%
inp = InputData(
    header    = f'{subdir} simulation -- tilt angle {45} -- nt {nt}',
    name      = name,    
    subdir    = subdir,
    f0        = f0,
    nt        = nt,
    nx        = nx,
    ny        = ny,
    dx        = dx,
    ne        = None,  
    waist     = waist,
    angle     = angle,
    yante     = yante,
    save_diag = save_diag, 
    mode      = mode, 
    b0        = B0,
)
# %%
if not size==1:

    t0 = time.time()
    # outp_gathered = scatterv_maxwell_v2(ne_map, dn, dny, ysteps, inp, root=root)
    outp_gathered = scatterv_maxwell_from_h5(inp, ne_lin, filename, t_start=0, t_end=None, root=0, fluct_lvl=0.05)
    # save the results
    if rank == root:
        print(outp_gathered.shape)
        print('time (s): ', time.time() - t0)
        np.save(inp.get_outp_dir() / 'ampl_phase.npy', outp_gathered)

else:
    t0 = time.time()
    ne = ne_lin * ( 1 + 0.02 * delta_ne[0])
    inp.ne = ne.astype(np.double)
    fw2d_wrapper(inp)
    print('time (s) : ', time.time() - t0)
# %%
if size == 1:
    outp      = OutputData(inp.name, subdir = inp.subdir)

    fig, ax = plt.subplots(figsize = (4, 4))

    Ez = outp.ez[int(inp.TFSF/2) : inp.nx + int(inp.TFSF/2), int(inp.TFSF/2) : inp.ny + int(inp.TFSF/2)]

    ax.pcolormesh(x, x, np.flip(Ez, axis = 1), cmap = 'jet')


    ax.pcolormesh(x, x, inp.ne, cmap = 'terrain', alpha = 0.3)


    fig, ax = plt.subplots(2, 1, figsize = (10, 6), sharex = True)
    ax[0].set_title(r'Amplitude of $E_z$')
    ax[0].plot(outp.ampl, c = 'dodgerblue')
    ax[1].plot(outp.phase / np.pi, c = 'dodgerblue')
    ax[0].set_ylabel('Amplitude [a.u.]')
    ax[1].set_ylabel('Phase [$\pi$]')
    ax[1].set_xlabel('Time [10 $\Delta t$]')
    plt.subplots_adjust(wspace = 0 )
    
    #%%
    ampl, phase = outp.doppler_data[:,0], outp.doppler_data[:,1]
    z = ampl * np.exp(1j * phase)
    #f,Pxx = welch(z, return_onesided=False)
    
    from scipy.signal import welch
    F, P = welch(z, return_onesided=False)
    combined = list(zip(F, P))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    sort_F, sort_P = zip(*sorted_combined)
    F, P = np.array(sort_F), np.array(sort_P)
    
    tf = np.fft.fft(z)
    nfft = tf.size
    f = np.linspace(-nfft/2, nfft/2 -1, nfft) / nfft
    Sk = np.fft.fftshift(np.abs(tf))

    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    
    ax[0].plot(f, 10 * np.log10(Sk / Sk.max()))
    ax[0].set_xlabel('f')
    ax[0].set_ylabel('[dBl]')
    
    ax[1].plot(F, 10 * np.log10(P / P.max()))
    ax[1].set_xlabel('f')
    ax[1].set_ylabel('[dBl]')
# %%

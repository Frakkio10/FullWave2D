#%%
from fullwave2d.core.wrapper import InputData, OutputData
from fullwave2d.core.parallel import launch_parallel_fw2d
from fullwave2d.main.turbulence_map import lin_prof, gaussian_map, SynthTurbMap


# %%
from mpi4py import MPI
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

ne_map = np.load('/home/forlacchio/FullWave2D/Density_map/57558_1024_slab.npy')

fig, ax = plt.subplots(1, 2, figsize = (10, 4))

im = ax[0].imshow(ne_map*1e-19, cmap = 'Blues', origin = 'lower')
fig.colorbar(im, ax = ax[0], orientation = 'vertical')

ne_tot = ne_map + 0.02 * ne_map * delta_ne

im = ax[1].imshow(ne_tot*1e-19, cmap = 'Blues', origin = 'lower')
fig.colorbar(im, ax = ax[1], orientation = 'vertical')
#%%
nt = 5000 # number of time steps
nx = int(1024)
ny = int(1024)
dx = 2.75e-4 # spatial resolution [m]
waist = 30 * dx
yante = 150 * dx
lmin = 0.51e-2
lmax = 1.4e-2
beta = 300*dx
angle = 15

map_args = (dx, nx, ny, lmin, lmax, beta)
delta_ne, spec, [x, y, kx, ky] = gaussian_map(*map_args)
synt = SynthTurbMap(map_args)

#%%
subdir='mpi_test_1024_angle15'
#angles = np.array([5, 10, 15, 20])
f0_arr = [45, 50, 55, 60]
simnames = [f'freq{f0:.1f}' for f0 in f0_arr]

def _make_inputs():

    inputs = []

    # global simulation parameters
    #f0 = 60e9 # probing frequency [Hz]
    nt = 5000 # number of time steps
    nx = int(1024)
    ny = int(1024)
    dx = 2.75e-4 # spatial resolution [m]
    waist = 30 * dx
    yante = int(ny/2 - 100) * dx
    lmin = 0.51e-2
    lmax = 1.4e-2
    beta = 20 * pi / 180
    angle = 15

    # define a place holder for the density map:
    # NOTE: The axes (x,y) of ne are reversed, because the source script
    # maxwell_2d_omode.c expects a 2d array of shape (ny, nx)

    # intitalize with linear background profile and
    # cutoff at halfway in the plasma
    map_args = (dx, nx, ny, lmin, lmax, beta)
    delta_ne, spec, [x, y, kx, ky] = gaussian_map(*map_args)
    synt = SynthTurbMap(map_args)

    dne = 0.05 * ne_map * delta_ne
    ne_tot = ne_map + dne
    
    for f0, name in zip(f0_arr, simnames):
        # input to the full wave script
        inp = InputData(
                        name = name,
                        f0 = f0*1e9,
                        nt = nt,
                        nx = nx,
                        ny = ny,
                        dx = dx,
                        ne = ne_tot,
                        waist = waist,
                        yante = yante,
                        angle = angle,
                        save_diag = True,
                        subdir=subdir
                        )

        inputs.append(inp)

    return inputs

def _run_parallel(inputs):
    
    t0 = time.time()  # time the computation

    if rank == root:
        print('Total number of inputs: ', len(inputs))


    # lauch parallelized full-wave simulations
    launch_parallel_fw2d(inputs, root=root)

    if rank == root:
        print('time (s): ', time.time() - t0)

#%%
if __name__ == '__main__':
    # run from a terminal with:    
    # mpiexec -n 5 python -m mpi4py run_parallel.py
    inputs = _make_inputs()
    _run_parallel(inputs)
    inputs = _make_inputs()
    axs = InputData.display_results(simnames, subdir=subdir)
# %%
from matplotlib import colors

fig, ax = plt.subplots(1, 4, figsize = (20, 6))
fig.suptitle('shot #57558 - twindow: [7.7, 7.8] s')

for i in range(0, 4):
    name = simnames[i]
    outp = OutputData(name, subdir)

    dPMLx = outp.ez.shape[1] - 1024
    dPMLy = outp.ez.shape[0] - 1024


    ez0 = outp.ez[int(dPMLy/2) : 1024 + int(dPMLy/2), int(dPMLx/2) : 1024 + int(dPMLx/2)]

    ez0 = np.flip(ez0, axis = 0)
    ez0 = np.flip(ez0, axis = 1)

    ax[i].set_title(r'$f_{prob}$ = %d GHz' %f0_arr[i])
    im = ax[i].imshow(ez0, aspect = 'equal', origin = 'lower', cmap = 'seismic', 
                    norm = colors.Normalize(vmin = -1, vmax = 1))
    fig.colorbar(im, ax = ax[i], orientation = 'horizontal', label = r'$E_z$ [a.u.]')
    im = ax[i].imshow(ne_tot*1e-19, cmap = 'Blues', origin = 'lower', alpha = 0.4)
    fig.colorbar(im, ax = ax[i], orientation = 'vertical', label = r'$n_{etot}$ $10^{19}$ [$m^{-3}$]')
# %%
outp = OutputData(simnames[0], subdir)
# %%
fig, ax = plt.subplots(2, 4, figsize = (20, 3))

for i in range(0, 4):
    outp = OutputData(simnames[i], subdir)
    ax[1,i].plot(outp.phase / pi, c = 'blue')
    ax[0,i].plot(outp.ampl, c = 'blue')
    ax[1,i].set_xlabel(r'10 $\Delta t$ [s]')


ax[0,0].set_ylabel('Ampl [a.u.]')
ax[1,0].set_ylabel(r'Phase [$\pi$]')

fig.subplots_adjust(hspace = 0)

# %%

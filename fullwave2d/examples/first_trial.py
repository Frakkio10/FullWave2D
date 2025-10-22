#%%
# # %matplotlib widget
import matplotlib.pyplot as plt
from matplotlib import animation, colors
import numpy as np
from numpy import pi
import time

from fullwave2d.main.turbulence_map import lin_prof, gaussian_map, SynthTurbMap
from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData

#%%
f0 = 60e9 # probing frequency [Hz]
nt = 5000 # number of time steps
nx = 400
ny = 400
dx = 2.75e-4 # spatial resolution [m]
lmin = 0.51e-2 # m
lmax = 1.4e-2 # m
beta = 20 * pi / 180 # rad
waist = 100 * dx
yante = 350*dx
angle = 15

map_args = (dx, nx, ny, lmin, lmax, beta)
delta_ne, spec, [x,y,kx,ky] = gaussian_map(*map_args)
synt = SynthTurbMap(map_args)

#display spectrum and the perturbation 
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax[0].set_title('Absolute k-spectrum')
ax[1].set_title('fluctuations')
synt.show_spectrum(ax = ax[0])
synt.show_fluct(ax = ax[1])

#%% Linear profile

#create the density map 
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ne = np.zeros([nx, ny]).astype(np.double) 
lin_prof(ne, f0, cut = 0.7, start = 0)
im = ax[0].imshow(ne*1e-19, cmap = 'Blues', origin='lower')

ne = np.flip(ne, axis = 1)
im1 = ax[1].imshow(ne*1e-19, cmap = 'Blues', origin='lower')
name0 = "linear profile - 1"
#creat the inputs 
inp0 = InputData(
                name = name0,
                f0 = f0,
                nt = nt, 
                nx = nx, 
                ny = ny, 
                dx = dx, 
                ne = ne,
                waist = waist, 
                yante = yante, 
                angle = angle, 
                save_diag = True, 
                mode = 'O'
)

#run the simulation
#t0 = time.time()
#fw2d_wrapper(inp0)
#print('time (s) :', time.time() + t0)
##
#%%

import plotly.graph_objs as go 
X = np.linspace(0, nx, nx)
Y = np.linspace(0, ny, ny)
fig = go.Figure(data = [go.Surface(z = ne*1e-19, x = X, y = Y, colorscale = 'Blues')])

fig.show()
# %% lineal profile + perturbation 

#create linear density profile 
ne = np.zeros([nx, ny]).astype(np.double) 
lin_prof(ne, f0, cut = 0.7, start = 0)
ne = np.flip(ne, axis = 1)

#create the perturbation using a Gaussian spectrum 
map_args = (dx, nx, ny, lmin, lmax, beta)
synt = SynthTurbMap(map_args)


#create the total density field 
dne =  0.02 * ne * np.flip(delta_ne, axis = 1)
ne_tot = ne + dne

name1 = "linear profile and perturbation - 1 "
#creat the inputs 
inp1 = InputData(
                name = name1,
                f0 = f0,
                nt = nt, 
                nx = nx, 
                ny = ny, 
                dx = dx, 
                ne = ne_tot,
                waist = waist, 
                yante = yante, 
                angle = angle, 
                save_diag = True, 
                mode = 'O'
)

#run the simulation
#t0 = time.time()
#fw2d_wrapper(inp1)
#print('time (s) :', time.time() + t0)
#

#%%
name0 = 'experimental profile'
name1 = 'experimental profile and perturbation '
input0 = inp0.load_pickle(name0)
input1 = inp1.load_pickle(name1)

#%%
output0 = OutputData(name0)
output1 = OutputData(name1)

# %%
fig, ax = plt.subplots(1, 2, figsize = (12, 4))


ax[0].set_title(r'Amplitude of $E_z$')
ax[0].plot(output0.ampl, c = 'dodgerblue')
ax[1].plot(output0.phase / pi, c = 'dodgerblue', label = 'linear prof')
ax[0].plot(output1.ampl, c = 'orangered')
ax[1].plot(output1.phase / pi, c = 'orangered', label = 'perturbed')
ax[0].set_ylabel('Amplitude [a.u.]')
ax[1].set_ylabel('Phase [$\pi$]')
ax[1].set_xlabel('Time [10 $\Delta t$]')
ax[0].set_xlabel('Time [10 $\Delta t$]')
ax[1].legend(loc = (0.8, 0.4))
plt.show()
# %%
fig, ax = plt.subplots(1, 2, figsize = (15, 8))
lvl = np.linspace(-1, 1, 11)

ax[0].set_title('Linear density profile')
im = ax[0].imshow(ne*1e-19, cmap = 'Blues', origin='lower')
fig.colorbar(im, ax = ax[0], orientation = 'horizontal', label = r'$n_{etot}$ $10^{19}$ $m^{-3}$')

ax[1].set_title('Total density profile')
im = ax[1].imshow(ne_tot*1e-19, cmap = 'Blues', origin='lower')
fig.colorbar(im, ax = ax[1], orientation = 'horizontal', label = r'$n_{etot}$ $10^{19}$ $m^{-3}$')

ax[0].set_xlim(0, 1024)
ax[0].set_ylim(0, 1024)
ax[1].set_xlim(0, 1024)
ax[1].set_ylim(0, 1024)
plt.show()

#display spectrum and the perturbation 
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
synt.show_spectrum(ax = ax[0])
synt.show_fluct(ax = ax[1])

plt.show()

#%%
fig, ax = plt.subplots(1, 2, figsize = (15, 8))
dPML = output0.ez.shape[0] - nx

fig.suptitle('Electric field propagation', fontsize = 20)
ez0 = output0.ez[int(dPML/2):nx+int(dPML/2), int(dPML/2):ny+int(dPML/2)]
ez1 = output1.ez[int(dPML/2):nx+int(dPML/2), int(dPML/2):ny+int(dPML/2)]

ax[0].set_title('Without Turbulence', fontsize = 16)
im = ax[0].imshow(np.flip(ez0),aspect='equal', origin = 'lower', cmap='seismic',
                  norm = colors.Normalize(vmin= -1, vmax =1))
fig.colorbar(im, ax = ax[0], orientation = 'horizontal', label = r'$E_z$ [a.u.]')
im2 = ax[0].imshow(ne*1e-19, cmap = 'Blues', origin='lower', alpha = 0.3)
fig.colorbar(im2, ax = ax[0], orientation = 'vertical', label = r'$n_{etot}$ $10^{19}$ $m^{-3}$')
#
#ax[0].set_xlim(0, 1024)
#ax[0].set_ylim(0, 1024)
#
ax[1].set_title('With Turbulence', fontsize = 16)

im = ax[1].imshow(np.flip(ez1),aspect='equal', origin = 'lower', cmap='seismic',
                     norm = colors.Normalize(vmin= -1, vmax =1))
fig.colorbar(im, ax = ax[1], orientation = 'horizontal', label = r'$E_z$ [a.u.]')
im2 = ax[1].imshow(ne_tot*1e-19, cmap = 'Blues', origin='lower', alpha = 0.3)
fig.colorbar(im2, ax = ax[1], orientation = 'vertical', label = r'$n_{etot}$ $10^{19}$ $m^{-3}$')

#ax[1].set_xlim(0, 1024)
#ax[1].set_ylim(0, 1024)

plt.show()

# %% Experimental density profile 

from francesco_feature.experimental_data import Experimental_profile

f0 = 60e9 # probing frequency [Hz]
nt = 5000 # number of time steps
nx = 1024
ny = 1024
dx = 2.75e-4 # spatial resolution [m]
lmin = 0.51e-2 # m
lmax = 1.4e-2 # m
beta = 20 * pi / 180 # rad
waist = 100 * dx
yante = 350*dx
angle = 15

map_args = (dx, nx, ny, lmin, lmax, beta)
delta_ne, spec, [x,y,kx,ky] = gaussian_map(*map_args)
synt = SynthTurbMap(map_args)

#display spectrum and the perturbation 
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax[0].set_title('Absolute k-spectrum')
ax[1].set_title('fluctuations')
synt.show_spectrum(ax = ax[0])
synt.show_fluct(ax = ax[1])
plt.show()


# %%
shot, twindow = 58333, [14.2, 14.8]
exp = Experimental_profile(shot, twindow)
ne_map = exp.get_density_map(nx, ny)
exp.visualize(plot_sep = True)
# %%
name0 = "experimental profile"
#creat the inputs 
inp0 = InputData(
                name = name0,
                f0 = f0,
                nt = nt, 
                nx = nx, 
                ny = ny, 
                dx = dx, 
                ne = np.flip(ne_map, axis = 1),
                waist = waist, 
                yante = yante, 
                angle = angle, 
                save_diag = True, 
                mode = 'O'
)

#t0 = time.time()
#fw2d_wrapper(inp0)
#print('time (s) :', time.time() + t0)
# %%
ne_map1 = np.flip(ne_map, axis = 1)
dne =  0.02 * ne_map1 * np.flip(delta_ne, axis = 1)
ne_tot = ne_map1 + dne

fig, ax = plt.subplots()

im2 = ax.imshow(ne_tot*1e-19, cmap = 'Blues', origin='lower')
fig.colorbar(im2, ax = ax, orientation = 'vertical', label = r'$n_{etot}$ $10^{19}$ $m^{-3}$')
#%%
name1 = "experimental profile and perturbation "
#creat the inputs 
inp1 = InputData(
                name = name1,
                f0 = f0,
                nt = nt, 
                nx = nx, 
                ny = ny, 
                dx = dx, 
                ne = np.flip(ne_tot, axis = 1),
                waist = waist, 
                yante = yante, 
                angle = angle, 
                save_diag = True, 
                mode = 'O'
)

#run the simulation
#t0 = time.time()
#fw2d_wrapper(inp1)
#print('time (s) :', time.time() + t0)
#
# %%
#%%
output0 = OutputData(name0)
output1 = OutputData(name1)

# %%
Rsep, Zsep = exp.get_equilibrium()
fig, ax = plt.subplots(1, 2, figsize = (15, 8))
dPML = output0.ez.shape[0] - nx

fig.suptitle('Electric field propagation', fontsize = 20)
ez0 = output0.ez[int(dPML/2):nx+int(dPML/2), int(dPML/2):ny+int(dPML/2)]
ez1 = output1.ez[int(dPML/2):nx+int(dPML/2), int(dPML/2):ny+int(dPML/2)]
ax[0].plot(Rsep, Zsep)
ax[0].set_title('Without Turbulence', fontsize = 16)
im = ax[0].imshow(np.flip(ez0),aspect='equal', origin = 'lower', cmap='seismic',
                  norm = colors.Normalize(vmin= -1, vmax =1))
fig.colorbar(im, ax = ax[0], orientation = 'horizontal', label = r'$E_z$ [a.u.]')
im2 = ax[0].imshow(ne_map*1e-19, cmap = 'Blues', origin='lower', alpha = 0.3)
fig.colorbar(im2, ax = ax[0], orientation = 'vertical', label = r'$n_{etot}$ $10^{19}$ $m^{-3}$')
#
#ax[0].set_xlim(0, 1024)
#ax[0].set_ylim(0, 1024)
#
ax[1].set_title('With Turbulence', fontsize = 16)

im = ax[1].imshow(np.flip(ez1),aspect='equal', origin = 'lower', cmap='seismic',
                     norm = colors.Normalize(vmin= -1, vmax =1))
fig.colorbar(im, ax = ax[1], orientation = 'horizontal', label = r'$E_z$ [a.u.]')
im2 = ax[1].imshow(np.flip(ne_tot, axis = 1)*1e-19, cmap = 'Blues', origin='lower', alpha = 0.3)
fig.colorbar(im2, ax = ax[1], orientation = 'vertical', label = r'$n_{etot}$ $10^{19}$ $m^{-3}$')

plt.show()
# %%

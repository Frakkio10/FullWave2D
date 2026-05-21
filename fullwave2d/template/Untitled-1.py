#%%
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
# %%
flname = '/home/FO278650/Zone_Travail/HWAK/simu_hwak/4096_4096_C1.0.h5'
it = 8
with h5py.File(flname, 'r', libver='latest', swmr=True) as fl:

    #Construct the real space grid
    Lx, Ly = fl['params/Lx'][()], fl['params/Ly'][()]
    Npx, Npy = fl['params/Npx'][()], fl['params/Npy'][()]
    Nx, Ny, Nxh, Nyh = int(Npx/3)*2, int(Npy/3)*2, int(Npx/3), int(Npy/3)
    X, Y = np.arange(0,Nx)*Lx/Nx, np.arange(0,Ny)*Ly/Ny 
    x, y = np.meshgrid(X, Y, indexing='ij')

    #Construct Fourier grid
    dkx, dky = 2*np.pi/Lx, 2*np.pi/Ly
    KX, KY = np.r_[np.arange(0,int(Nx/2)+1)*dkx, np.arange(-int(Nx/2)+1, 0)*dkx],  np.arange(0, int(Ny/2)+1)*dky
    kx, ky = np.meshgrid(KX, KY, indexing='ij')
    ksqr=kx**2+ky**2

    # Linear parameters
    C, kap = fl['params/C'][()], fl['params/kap'][()]
    nu, D = fl['params/nu'][()], fl['params/D'][()]
    uk = fl['fields/uk'][it]

    print('C = %.1f, kap = %.1f, nu = %.1e, D = %.1e' %(C, kap, nu, D))
    print('Lx = %.1f, Ly = %.1f, Npx = %.1f, Npy = %.1f' %(Lx, Ly, Npx, Npy))
ubar = np.real(np.fft.irfft(1j*kx[:Nxh+1, 0]*uk[0, :Nxh+1 , 0], norm='forward'))
nbar = np.real(np.fft.irfft(uk[1, :Nxh+1, 0], norm='forward'))

ksqr = kx**2+ky**2
Ezf =  ksqr[:Nxh,0] * abs(uk[0,:Nxh,0])**2
q_zf = np.sum(np.abs(np.sqrt(ksqr[:Nxh,0])) * Ezf) / np.sum(Ezf)

# %%
with h5py.File(flname, 'r', libver='latest', swmr=True) as fl:
    nk = fl['fields/density/nk'][10:200]

dn = np.fft.irfft2(nk, norm='forward')

# %%
n0 = - kap * (x[-1500:,-1500:]- Lx )
dn_over_n = dn[:,-1500:,-1500:] / (n0[np.newaxis, -1500:,-1500:] + dn)
# rms_2d = np.sqrt(np.mean((dn[:,-1500:,-1500:] / (n0 + dn))**2, axis=0))  # (Nx, Ny)
# rms_y = np.nanmean(rms_2d, axis=1)            # shape (Nx,)

# %%
fig, ax = plt.subplots(figsize = (8, 4))
axdn = ax.twinx()

# im = ax.pcolormesh(X, np.linspace(10, 100, 90), rms  , cmap = 'coolwarm')
im = ax.pcolormesh(x[-1500:,-1500:], y[-1500:,-1500:], rms_2d - rms_2d.mean(axis = 0)  , cmap = 'coolwarm', shading = 'auto')

axdn.plot(X[-1500:], rms_y, c = 'k')
fig.colorbar(im, ax = ax)
# %%
ntot = n0 + dn[0,-1500:,-1500:]

fgi, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[0].pcolormesh(x[-1500:,-1500:], y[-1500:,-1500:], ntot, cmap = 'terrain', shading = 'auto')
dn_over_n = dn[0,-1500:,-1500:] / n0

axdn = ax[0].twinx()
axdn.plot(X[-1500:], dn_over_n.mean(axis = 0), c = 'k', lw = 2)


nlin = n0 * 30e17 + 3e17
ne_tot = nlin* (1 + 0.2 * dn[0, -1500:,-1500:] / dn[0, -1500:,-1500:].max())

ax[1].pcolormesh(x[-1500:,-1500:], y[-1500:,-1500:], ne_tot, cmap = 'terrain', shading = 'auto')
dn_over_n = dn[[0,10], -1500:,-1500:] / nlin
axdn = ax[1].twinx()
axdn.plot(X[-1500:], dn_over_n.mean(axis = 0) *1e19, c = 'k', lw = 2)

#%%
rms = np.zeros([40, 1500, 1500])
for i in range(0, 40):
    dn_over_n = dn[i, -1500:,-1500:] / nlin
    rms[i] = np.sqrt(dn_over_n.mean(axis = 0) ** 2)

# %%

fig, ax = plt.subplots(figsize = (7, 4))
ax.pcolormesh(x[-1500:, -1500:], y[-1500:, -1500:], rms.mean(axis = 0).T, cmap = 'RdBu_r')

axn = ax.twinx()
axn.plot(X[-1500:], rms.mean(axis = 0).mean(axis = 0) * 10, c = 'k', lw = 2)
# %%

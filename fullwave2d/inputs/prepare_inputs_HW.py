#%%

import numpy as np 
import matplotlib.pyplot as plt 
import h5py as h5
import scipy
from scipy.ndimage import rotate

# %%
def get_rho_s(Te, B):
    mi = scipy.constants.m_p + scipy.constants.m_n #kg
    q = scipy.constants.e                          #C 
    rho_s = np.sqrt(Te * q * mi) / (q * B)
    return rho_s

class HW_for_FW2D():
    
    def __init__(self, filename):
        
        self.filename = filename
        
    def compose_ne(self, nlin, delta_n, fluct_level, X, xc, nc, n_off):

        n_norm = nlin + fluct_level * delta_n
        idx_c = np.argmin(np.abs(X - xc))
        idx_r = np.argmin(np.abs(X - X[-1]))

        n_c_norm = np.mean(n_norm[idx_c, :])
        n_r_norm = np.mean(n_norm[idx_r, :])

        n0 = (nc - n_off) / (n_c_norm - n_r_norm)
        ne_map = n0 * n_norm + n_off

        return ne_map 
        
    def read_file_parameters(self):
        
        with h5.File(self.filename, 'r', libver='latest', swmr=True) as fl:
            Lx, Ly = fl['params/Lx'][()], fl['params/Ly'][()]
            Npx, Npy = fl['params/Npx'][()], fl['params/Npy'][()]
            
            # Linear parameters
            C, kap = fl['params/C'][()], fl['params/kap'][()]
            nu, D = fl['params/nu'][()], fl['params/D'][()]

            # time array
            t = fl['fields/t'][:]

        Nx, Ny, Nxh, Nyh = int(Npx/3)*2, int(Npy/3)*2, int(Npx/3), int(Npy/3)
        X, Y = np.arange(0,Nx)*Lx/Nx, np.arange(0,Ny)*Ly/Ny 

        self.grid          = [X, Y, Lx, Ly]
        self.linear_params = [C, kap, nu, D]
        self.time          = t
        
        return [X, Y, Lx, Ly], [C, kap, nu, D]
    
    def read_file_fields(self, it, return_all = False):
        
        with h5.File(self.filename, 'r', libver='latest', swmr=True) as fl:
            #reading fields
            uk = fl['fields/uk'][it]
        
        #Computing the Fourier transform og n at time step it 
        n = np.fft.irfft2(uk[1,], norm='forward')
        
        if return_all:
            phi = np.fft.irfft2(uk[0,], norm='forward') 
            return n, phi  
        else:
            return n
        
    def prepare_ne_map(self, n, it, fluct_level, Te = 300, B = 3.2, xc = 0.4, nc = 4e19, n_off=1.5e19, **kwargs):
        
        dn = self.read_file_fields(it)
        rho_s = get_rho_s(Te, B)
        
        x, y = np.meshgrid(self.grid[0], self.grid[1], indexing = 'ij')
        dn_over_n = dn / n

        X, Y, Lx, Ly= self.grid[0] * rho_s, self.grid[1] * rho_s, self.grid[2] * rho_s, self.grid[3] * rho_s
        x, y = np.meshgrid(X, Y, indexing = 'ij')
        
        nlin = -self.linear_params[1] * (x - Lx)
        delta_n = dn_over_n * nlin
        
        xc = xc * Lx
        
        ne_map = self.compose_ne(nlin, delta_n, fluct_level, X, xc, nc, n_off)
        ne_map = ne_map.T.astype(np.double)
        ne_map = rotate(ne_map, angle=0, reshape=False, order=1,mode='nearest')
        
        self.ne_map      = ne_map
        self.x, self.y   = x, y
        self.t           = it
        self.xc, self.nc = xc, nc
        
        return x, y, ne_map
    
    def plot_ne(self, x = None, y = None, ne_map = None, ax = None, fig = None, show_all = False, **plot_kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(figsize = (5, 4))
        if ne_map is None:
            x, y, ne_map = self.x, self.y, self.ne_map
            ax.set_xlabel('$x$ [m]', fontsize = 12)
            ax.set_ylabel('$y$ [m]', fontsize = 12)
        
        qd_n = ax.pcolormesh(x, y, ne_map.T, vmin=np.min(ne_map), vmax = np.max(ne_map), rasterized=True, **plot_kwargs)
        fig.colorbar(qd_n, label='$n(x,y)$', pad=0.05)
        
        if show_all:
            axi = ax.twinx()
            nr = np.mean(ne_map.T, axis=-1)
            axi.plot(self.x[:,0], nr, lw=6, color='w', alpha=0.9)
            axi.plot(self.x[:,0], nr, lw=4, color='k', alpha=0.9)
            axi.plot(self.xc, self.nc, 'Xr', markersize = 8)
            axi.set_ylim([0, np.max(nr)])
            axi.set_ylabel('$n_r(x,t)$')


        ax.set_title(f'$n(x,y)$ -- $C={self.linear_params[0]}$ @ $t={self.t:.1f}$', fontsize = 12, pad=10)

# %%

if __name__ == '__main__':
    flname = '/home/forlacchio/FullWave2D/HWAK/hwak_simu/test_outC0.01_t500s.h5'

    HW_inp = HW_for_FW2D(flname)
    #%%
    [X, Y, Lx, Ly], [C, kap, nu, D] = inputs.read_file_parameters() 
    _x, _y = np.meshgrid(X, Y, indexing = 'ij')

    inputs.n = - kap * (_x - Lx)

    x, y, ne_map = inputs.prepare_ne_map(inputs.n, 450, 1)

    inputs.plot_ne( cmap = 'terrain')

    # %%

    mi = scipy.constants.m_p + scipy.constants.m_n #kg
    q = scipy.constants.e 

    Omega_i = 2*q * 3.2 / mi
    # %%
    def trial(inp):
        [X, Y, Lx, Ly], [C, kap, nu, D] = inp.read_file_parameters() 
        inp.n = - kap * (_x - Lx)

        print(C, nu)
        
    
        
# %%
    filename = '/home/forlacchio/FullWave2D/HWAK/hwak_simu/test_outC0.01_t500s.h5'

    with h5.File(filename, 'r', libver='latest', swmr=True) as fl:
        #reading fields
        uk = fl['fields/uk'][100]
        t = fl['fields/t'][:]
# %%

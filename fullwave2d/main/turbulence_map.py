import h5py
import numpy as np
from numpy import pi, sin, cos, tanh
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import map_coordinates
from pathlib import Path
from fullwave2d import definitions
import scipy.io as sio

from .misc import tanh_transition, find_nearest


def rms(arr):
    """ Returns the root mean squared of an array """
    N = arr.size # total number of elements
    rms = np.sqrt(np.sum(arr**2) / N)
    return rms

def gaussian_map(dx, nx, ny, lmin, lmax, beta, dkx=0, dky=0):
    """
    Create a synthetic fluctuation map according
    to a Gaussian k 2d spectrum.
    """
    # for reproducibility:
    np.random.seed(12345)

    kx_max = 1 / 2 / dx
    ky_max = 1 / 2 / dx
    # kx_max = pi / dx
    # ky_max = pi / dx

    kx, kx_min = np.linspace(-kx_max, kx_max, nx, retstep=True)
    ky, ky_min = np.linspace(-ky_max, ky_max, ny, retstep=True)

    kx = kx - dkx
    ky = ky - dky

    KX, KY = np.meshgrid(kx, ky)

    # random phase
    phi = 2 * pi * np.random.random((ny, nx)) - pi

    # compute absolute gaussian k spectrum:
    a = (KX * cos(beta) - KY * sin(beta))**2
    b = (KX * sin(beta) - KY * cos(beta))**2
    abs_spec = np.exp(-1/8 * (a * lmax**2 + b * lmin**2))
    abs_spec *= lmin * lmax / 8 / pi

    # add the complex phase
    spec = abs_spec * np.exp(-1j * phi)

    # compute fluctuations from spectrum via
    # inverse FT:
    Z = np.fft.ifft2(np.fft.fftshift(spec))

    x = dx * np.arange(nx)
    y = dx * np.arange(ny)

    # print(x)
    # print(y)

    # density perturbations
    delta_ne = np.real(Z)

    # normalization to rms value of 1
    N = delta_ne.size # total number of elements
    rms = np.sqrt(np.sum(delta_ne**2) / N)
    delta_ne /= rms

    # rms = np.sqrt(np.sum(delta_ne**2) / N)
    # print('rms of the normalized turbulence field: ', rms) # unity
    # print(np.var(delta_ne)**0.5) # approx. unity
    # print(np.mean(delta_ne)) # approx. 0

    return delta_ne, spec, [x,y,kx,ky]

def generate_turb_from_spec(spec):

    """

    """
    # for reproducibility:
    np.random.seed(12345)

    # random phase
    phi = 2 * pi * np.random.random(spec.size) - pi

    # add the complex phase
    spec = np.abs(spec) * np.exp(-1j * phi)

    # compute fluctuations from spectrum via
    # inverse FT:
    Z = np.fft.ifft(spec)

    # x = dx * np.arange(nx)
    # y = dx * np.arange(ny)

    # print(x)
    # print(y)

    # density perturbations
    delta_ne = np.real(Z)

    # normalization to rms value of 1
    N = delta_ne.size # total number of elements
    rms = np.sqrt(np.sum(delta_ne**2) / N)
    delta_ne /= rms

    # rms = np.sqrt(np.sum(delta_ne**2) / N)
    # print('rms of the normalized turbulence field: ', rms) # unity
    # print(np.var(delta_ne)**0.5) # approx. unity
    # print(np.mean(delta_ne)) # approx. 0

    return delta_ne

def get_ncrit(f0, angle=0.0):
    """
    Critical density for vacuum-frequency f0 [Hz] for
    O/X-mode and incidence angle [degrees].
    """
    # eps0 * m_e * (2 pi)² /e² [SI units] = 0.012404426
    return f0**2 * 0.012404426 * np.cos(np.deg2rad(angle))

def get_ncrit_X(f0, b0, angle=0.0):
    """
    Compute cutoff densities nR, nL for X-mode according to the cutoff
    frequencies as found e.g. here:
    https://en.wikipedia.org/wiki/Electromagnetic_electron_wave

    Args:
        - f0 (float): vaccum-frequency in [Hz]
        - b0 (float): magnetic field in [Tesla]
        - angle (degrees): incidence angle # NOTE: is a factor of cos(angle) correct for X-mode as well?
    Returns:
        - (nR, nL) (float): lower and upper cut-off densities
    """
    from scipy.constants import m_e, e, epsilon_0

    # freq in rad/s:
    om = 2*pi* f0

    # cyclotron freq
    omc = e * b0 / m_e

    _R = (2*om - omc)**2 - omc**2
    _L = (2*om + omc)**2 - omc**2

    prefac = m_e * epsilon_0 / 4 / e**2
    angle_fac = np.cos(np.deg2rad(angle))

    nR = prefac * _R * angle_fac
    nL = prefac * _L * angle_fac

    return nR, nL

def get_cutoff_X_freq(ne, b0):
    """
    (Not needed yet)
    Returns the cutoff frequencies omR, omL for X-mode
    for a given density ne and magnetic field b0 (SI units).
    """
    from scipy.constants import m_e, e, epsilon_0

    # cyclotron freq
    omc = e * b0 / m_e
    omp = e * np.sqrt(ne / m_e / epsilon_0)
    _ = np.sqrt(omc**2 + 4*omp**2)
    omR = 0.5 * ( omc + _)
    omL = 0.5 * (-omc + _)

    return omR / 2/pi, omL /2/pi

def get_cutoff(f0, n0, angle=0.0, mode='O', b0=0):
    """
    Given vacuum-frequency f0 [Hz] and density profile n0 [m⁻³] array,
    get the array index and element nearest to the nominal cutoff.
    """
    if mode=='O':
        icrit, ncrit = find_nearest(n0, get_ncrit(f0, angle=angle))
    elif mode=='X':
        nR, nL = get_ncrit_X(f0, b0, angle=angle)
        icrit, ncrit = find_nearest(n0, nL) # choose the upper limit

    return icrit, ncrit

def lin_prof(ne, f0, cut=0.5, start=0, mode='O', b0=0, angle=0):
    """
    Compute a linear linearly increasing
    density profile along x with custom turning point.
    Args:
        ne: (ny x nx) 2d array in which to store the profile
        f0: probing frequency in Hz
        dx: grid spacing
        cut: position of the turning point as a fraction
             of the computational domain
        start: position of the vacuum-plasma boundary as a fraction
               of the computational domain
    Returns:
        i_crit (int) : the index along axis x of the nominal cutoff
        n_crit (float): density in [m⁻³] at cutoff
    """
    nx = ne.shape[1]

    i_start = int(nx * start)
    # critical density in [m⁻³]

    if mode=='O':
        n_crit = get_ncrit(f0, angle)
    elif mode=='X':
        nR, nL = get_ncrit_X(f0, b0, angle=angle)
        n_crit = nL

    i_crit = int(nx * cut)

    for i in range(nx):
        if i >= i_start:
            ne[:,i] = (i - i_start) / i_crit * n_crit

    return i_crit, n_crit

def get_slice(arr, sl_extent, arr_extent, retindices=False):
    """
    Take a slice of a 2D array according to the extent arguments.

    Args:
        - arr (ndarray): original 2D array from which the slice is taken
        - sl_extent (float tuple): extent of the slice with respect to the
         array extent, in this order: [left, right, bottom, top]
        - arr_extent (tuple or None): Array extent as a reference for the
        slice extent. If None (the default), the array dimensions are assumed, i.e. sl_extent then correspond to the indices along each axis.
        - retindices (bool, optional):
            whether to return also the indices of the extent wrt original array
    Returns:
        - arr_slice (ndarray): Slice of the original array whose dimensions are determined by the extent arguments.
        - retindices (array-like): indices corresponding the slice extent. Only returned if specified by `retindices` argument.

    """
    Ny, Nx = arr.shape

    left, right, bottom, top = sl_extent

    x0, x1, y0, y1 = arr_extent

    # get the indices corresponding to left,right,top,bottom
    # according to the array extents along x and y:
    ileft   = int(Nx * (left - x0)   / (x1 - x0))
    iright  = int(Nx * (right - x0)  / (x1 - x0))
    ibottom = int(Ny * (bottom - y0) / (y1 - y0))
    itop    = int(Ny * (top - y0)    / (y1 - y0))

    dym, dyp, dxm, dxp = 0,0,0,0

    if ibottom < 0:
        dym = abs(ibottom)
    if itop >= Ny:
        dyp = (itop - Ny +1)
    if ileft < 0:
        dxm = abs(ileft)
    if iright >= Nx:
        dxp = (iright - Nx +1)

    extended_arr = np.pad(arr, ((dym, dyp), (dxm, dxp)), mode='edge')
    arr_slice = extended_arr[ibottom:itop, ileft:iright]

    if retindices:
        return arr_slice, [ileft, iright, ibottom, itop]
    else:
        return arr_slice

def get_simbox_size(sl_extent, dx, dy=None):
    """
    Compute the size (ny, nx) of the simulation domain,
    given the extent (in data coordinates) of the slice used as input
    and the spatial resolution dx (and dy).

    Args:
        - sl_extent (float tuple) :
            [left, right, bottom, top] extent
        - dx (float):
            spatial resolution used in the full-wave simulation
        - dy (float, optional):
            spatial resolution along y, by default same as dx
    Returns:
        - (ny, nx) (int, int): size of the simulation box

    """

    if dy is None:
        dy = dx
    left, right = sl_extent[0], sl_extent[1]
    bottom, top = sl_extent[2], sl_extent[3]

    nx = (right - left) / dx
    ny = (top - bottom) / dy

    return int(ny), int(nx)

def transform_resolution(arr, ny, nx):
    """
    Converts the input array of shape (nys, nxs) to the
    new resolution defined by (ny, nx) by spline interpolation.
    Note that the ascpect ratio of the old array is only conserved
    if nys/nxs = ny/nx.
    """
    nys, nxs = arr.shape
    spline = RectBivariateSpline(np.linspace(0, ny, nys), np.linspace(0, nx, nxs), arr)

    new_grid_x = np.arange(nx)
    new_grid_y = np.arange(ny)

    new_arr = spline(new_grid_y, new_grid_x, grid=True)

    return new_arr

def polar2cartesian(theta, r, polar_im, x, y, order=3):
    """
    Converts a 2d data set from polar (theta, r) to cartesian (x, y)
    coordinates via interpolation.
    Args:
        - theta  (array-like):  the poloidal coordinates
        - r      (array-like):  the radial   coordinates
        - polar_im  (ndarray):  the data defined on the polar mesh (theta, r)
        - x      (array-like):  horizontal cartesian axis to project the data on
        - y      (array-like):  vertical   cartesian axis to project the data on
        - order         (int):  interpolation order (default is cubic interp.)
    Returns:
        - map_coords (ndarray):
    """
    # rotate the data by dtheta (=first element of theta):
    Nt = len(theta)
    shift_theta = int(theta[0]/2/pi * Nt)
    _polar_im = np.roll(polar_im, shift=shift_theta, axis=0)

    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X*X+Y*Y)
    new_t = np.arctan2(Y, X)

    _theta = np.linspace(-pi, pi, Nt)

    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(_theta, np.arange(Nt), bounds_error=False)

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    map_coords = map_coordinates(_polar_im, np.array([new_it, new_ir]),
                           order=order, mode='constant', cval=np.nan).reshape(new_r.shape)

    return map_coords

def tanh_edge_drop(arr, axes=None, a=0.05, b=0.01, edges=None):
    """
    Ramp down the values of an array along the given axes.
    The drop is achieved by multiplication with a tanh function.

    Args:
        - arr (ndarray):
            original array whose values shall be ramped down towards the edges
        - axes (int or tuple of ints):
            the axes along which to apply the drop
        - a (float):
            Tanh symmetry point (as a fraction of number of elements
            along the axis). Same is applied to the other side, i.e. at (1-a).
        - b (float) steepness parameter of the tanh
        - edges (bool, array-like):
            Specifiecies, for each axis, Whether to apply on left, right or on both edges. None (default) applies to both edges for all axes.
    Returns:
        - new_arr (ndarray):
            copy of the input array but with values dropping to zero at the edges

    Example usage:

    a = np.ones((300, 200))
    b = tanh_edge_drop(a)
    fig, ax = plt.subplots()
    ax.imshow(b)
    plt.show()

    """

    if axes is None:
        axes = np.arange(arr.ndim)
    elif type(axes) != list:
        axes = [axes]

    if edges is None:
        edges = np.array([[True, True] for i in range(arr.ndim)])

    new_arr = np.copy(arr)

    for ax in axes:
        n = arr.shape[ax]
        x = np.arange(n)

        if edges[ax][0]:
            mask_left  = tanh_transition(x, 0.0, 1.0, n *  a    , n * b)
        else:
            mask_left = np.ones_like(x)
        if edges[ax][1]:
            mask_right = tanh_transition(x, 1.0, 0.0, (n-1) * (1-a) , n * b)
        else:
            mask_right = np.ones_like(x)
        mask = mask_left * mask_right

        #mask[0]  = 0.0
        #mask[-1] = 0.0
        shp = np.ones(arr.ndim).astype(int)
        shp[ax] = n
        mask = mask.reshape(shp)

        new_arr = new_arr * mask

    return new_arr



class SynthTurbMap():

    def __init__(self, map_args,**kwargs):

        delta_ne, spec, [x,y,kx,ky] = \
        gaussian_map(*map_args)

        self.delta_ne = delta_ne
        self.spec = spec
        self.x = x
        self.y = y
        self.kx = kx
        self.ky = ky

        self.unit = kwargs.get('unit', 'cm')

    def uf(self):
        """
        Returns the unit conversion factor
        meter/centimeter for the plots
        """
        return 1e-2 if self.unit == 'cm' else 1

    def show_spectrum(self, ax=None):
        """
        Display the image of the absolute k
        spectrum
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        uf = self.uf()

        kx_max = np.max(self.kx)
        ky_max = np.max(self.ky)

        im = ax.imshow(np.abs(self.spec), extent= uf * np.array([-kx_max, kx_max, -kx_max, kx_max]), cmap='plasma', interpolation='none', origin='lower')
        fig.colorbar(im, ax=ax, label='abs. PSD [arb. units]')
        ax.set_xlabel('$k_x$ [{}'.format(self.unit) +'$^{-1}$]')
        ax.set_ylabel('$k_y$ [{}'.format(self.unit) +'$^{-1}$]')

        return ax

    def show_fluct(self, ax=None):
        """
        Display the image of the fluctuations
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        uf = self.uf()
        x,y = self.x, self.y

        im = ax.imshow(self.delta_ne, extent= 1/uf * np.array([x[0], x[-1], y[0], y[-1]]), cmap='seismic', interpolation='none', origin='lower')
        fig.colorbar(im, ax=ax, label='fluctuation $\delta n$')
        ax.set_xlabel('$x$ [{}]'.format(self.unit))
        ax.set_ylabel('$y$ [{}]'.format(self.unit))


        return ax

class GyselaMap():
    """
    A class grouping the methods needed to load raw data
    from GYSELA simulations, extract density profiles and create
    2D cartesian maps from the polar data.
    To get the e.g. density map, do:
    gysmap = GyselaMap()
    gysmap.load(fname, init_fname)
    gysmap.generate_cartesian_maps()
    ne = gysmap.ne

    """

    def __init__(self, dtheta=0.0, suppr_edge_fluct=True):
        """
        Args:
            - dtheta (float):
            initial poloidal clock-wise rotation in [rad] (dtheta = 0 is
            the conventional view where y is the vertical axis and the LFS
            is on the r.h.s.)
            - suppr_edge_fluct (bool): whether to reduce high edge fluctuations
            outside of the confined region, see GyselaMap.edge_fluct_drop()
        """
        self.dtheta = dtheta
        self.suppr_edge_fluct = suppr_edge_fluct

    def load(self, fname, init_fname, nprof_choice='gys'):
        """
        Loads all the relevant data from the the HDF5 files, where the first
        filename corresponds to a potential realization at a given timestep,
        while the second one is for the background profile (assumed to be approx. constant.

        Args:
            - fname (str): path to a 'Phi2D_dXXXXX.h5' file
            - init_fname (str): path to corresponding 'init_state_rXXX.h5' file
            - nprof_choice (str): 'gys' or 'exp', affects the shape of
                                   the density profile (see get_denormalized_nprof())
        """

        #first, gather information from the init_state file
        f = h5py.File(init_fname, 'r')
        rg, rhostar, thetag = np.array(f['rg']), np.array(f['rhostar']), np.array(f['thetag'])
        Te0 = np.array(f['Te0'])
        # ne0 = np.array(f['ne0']) # e⁻ density imposed in GYSELA (not needed)
        limiter = np.array(f['LIMITER_shape'])
        iota = np.array(f['iota'])

        # 2D data including fluctuations:
        f2 = h5py.File(fname, 'r')
        phi = np.array(f2['Phirth'])
        phi0 = np.array(f2['Phirth_n0'])
        dphi = phi - phi0 # this is proportional to density fluctuations

        # convert potential to density fluctuations
        dn = np.copy(dphi)
        dn[:] /= (Te0) # Te0 depends only on r

        if self.suppr_edge_fluct:
            dn = self.edge_fluct_drop(dn)

        self.phi  = phi
        self.phi0 = phi0
        self.Te0 = Te0
        self.dn = dn # relative density fluctuations

        # density profile in [m⁻³] to be supplied to full-wave sim.:
        n0 = np.load(definitions.GYS_DIR / 'nprof_fullwave.npy')
        self.n0 = n0 # radial eq. density profile after de-normalization

        self.theta = np.linspace(0, 2*pi, 1025, endpoint=False) # poloidal coord.
        self.theta -= self.dtheta # initial poloidal rotation
        self.r = rg * rhostar # radial coord. normalized to a
        self.extent_rth = [self.r[0], self.r[-1],
                           self.theta[0], self.theta[-1]] # extent of polar coordinates (rmin, rmax, theta_min, theta_max)

        self.limiter =limiter
        self.iota = iota

        # keep these for safety:
        self.fname = fname
        self.init_fname = init_fname

    def generate_cartesian_maps(self, Nx=1500):

        """
            Args:
                - dtheta (float):
                    poloidal clock-wise rotation (dtheta = 0 is
                    the conventional view where y is the vertical axis)
        """

        # establish x and y arrays
        rmax = np.amax(self.r)
        x, dx = np.linspace(-rmax, rmax, Nx, retstep=True)
        y = np.copy(x)
        self.x, self.y, = x, y
        self.extent = [-rmax, rmax, -rmax, rmax]

        # mark a reference point for testing purposes:
        # self.dn[0:10, 200:220] = 0.2
        # self.dn[1025//4:1025//4+10, 200:220] = 0.2

        # conversion of dn to cartesian coords: (r,theta) -> (x, y)
        dn_cart = polar2cartesian(self.theta, self.r, self.dn, self.x, self.y, order=3)

        # flip around x=y axis (so that limiter at the bottom and LFS on the r.h.s)
        dn_cart = np.flip(dn_cart, axis=0)
        dn_cart = np.flip(dn_cart, axis=1)

        # remove np.nans at the edges
        dn_cart[np.isnan(dn_cart)] = 0.0

        # get a cartesian (x,y) map of background density:
        # invariance along theta, so two points suffice (lin. interpolation):
        # _theta = np.linspace(0, 2*pi, 2)
        n0_rth = np.reshape(self.n0, (1, -1))
        n0_rth = n0_rth.repeat(self.theta.size, axis=0)
        n0_cart = polar2cartesian(self.theta, self.r, n0_rth, self.x, self.y)

        # replace nans by zeros:
        n0_cart[np.isnan(n0_cart)] = 0.0

        # Finally, add fluctuations to the background density
        ne = np.copy(n0_cart)
        ne = ne * (1 + dn_cart)
        ne[ne<0] = 0.0 # negative densities are harmful

        ### assign the relevant attributes to this instance ###
        self.dn_cart = dn_cart # (x,y) map of fluctutations normalized to radial backgr. profile
        # self.n0_smooth = n0_smooth # radial eq. density profile before de-norm
        self.n0_rth  = n0_rth  # (r,theta) map of backgr. density
        self.n0_cart = n0_cart # (x,y) map of backgr. density
        self.ne = ne # full (x,y) map (backgr. + fluctuations)

    def get_gyselamap(fname=None, init_fname=None, Nx=1500, dtheta=0, suppr_edge_fluct=True, nprof_choice='gys'):
        """
        Convenience function that sets up and returns an instance
        of GyselaMap().
        """

        if fname is None:
            p = definitions.DATA_DIR / 'ne_fluct/gysela/Phi2D/'
            fnames = [x for x in sorted(p.iterdir()) if x.suffix == '.h5']
            fname = fnames[-1]
        if init_fname is None:
            init_fname = Path(definitions.DATA_DIR) / 'ne_fluct/gysela/init_state/init_state_r000.h5'

        gysmap = GyselaMap(dtheta, suppr_edge_fluct)
        gysmap.load(fname, init_fname, nprof_choice)
        gysmap.generate_cartesian_maps(Nx=Nx)

        return gysmap

    # Deprecated name: get_example_map --> get_gyselamap:
    get_example_map = get_gyselamap

    def get_ne_slice(sl_extent, ny, nx, attr='ne', **gysela_args):
        """
        Shortcut for setting up the GyselaMap (pass keyword arguments to
        GyselaMap.get_gyselamap()) and extracting the desired slice based on
        sl_extent and full-wave simulation domain size (ny,nx) (see get_slice()).
        """
        gysmap = GyselaMap.get_gyselamap(**gysela_args)

        _sl = get_slice(getattr(gysmap, attr), sl_extent, gysmap.extent)
        # transform slice into shape (ny, nx):
        sl = transform_resolution(_sl, ny, nx)
        # ramp down density to zero at simulation boundaries
        sl = tanh_edge_drop(sl, axes=[0,1], a=0.03, b=0.015, edges=[[True,True], [True, False]])

        return sl

    def edge_fluct_drop(self, dn, r_edge=1.1, b=0.02):
        """Suppress edge fluctuations for r/a > r_edge by modified
        tanh with steepness parameter b."""
        a = (1.3 - r_edge) / 1.3
        dn_new = tanh_edge_drop(dn, axes=[1], a=(1.3-1.1)/1.3, b=0.02, edges=[[], [False,True]])
        return dn_new

    def show_limiter(self, ax):
        from fullwave2d.visualize import plot_limiter
        lim = polar2cartesian(self.theta, self.r, self.limiter, self.x, self.y, order=3)
        lim = np.flip(lim, axis=0)
        lim = np.flip(lim, axis=1)
        im = plot_limiter(lim, ax, self.extent)
        return im

    def show_poloidal_plot(self, ax, diag='ne'):

        if diag in ['ne', 'n0_cart', 'dn_cart']:

            if diag =='ne':
                imst = get_default_imstyle()
                norm = colors.Normalize(vmin=-ma, vmax=ma)
                imst4.imshow_kwargs['norm'] = norm
                imst4.imshow_kwargs['extent'] = [x[0], x[-1], y[0], y[-1]]
                imst4.get_im(turbmap, ax)
                imst4.add_colorbar(ax)
                ax.set_xlabel('$x/a$')
                ax.set_ylabel('$y/a$')

    def B_field(r, theta, q, B0=1, R0=2.4, meshgrid=True):
        """
        Args:
            - theta: poloidal angle in [deg]
        """
        if meshgrid:
            mr, mtheta = np.meshgrid(r, theta)
        else:
            mr, mtheta = r, theta

        if 1 in mr.shape:
            mr = mr.flatten()
        if 1 in mtheta.shape:
            mtheta = mtheta.flatten()

        R = R0 + mr * cos(np.deg2rad(mtheta))
        Btor = B0 * R0 / R
        Bpol = Btor * mr / q / R0

        return Btor, Bpol

    def Efield(rho, Phi, irhostar=250):
        """
        Compute electric field in normalized GYSELA units from radial profile of electrostatic potential.

        Args:
        - rho:  normalized radial coordinate r/a
        - Phi: potential in GYSELA normalized units
        - irhostar:  inverse normalized Larmor radius 1/rhostar
        """
        from scipy.interpolate import splev, splrep
        rg = rho * irhostar
        spl = splrep(rg, -Phi)
        return splev(rg, spl, der=1)

    def Efield_SI(r, Phi, T0=None):
        """
        DEPRECATED
        Compute electric field in SI unit from radial profile of GYSELA
        electrostatic potential.

        Args:
            - r:   radial coordinate in [m]
            - phi: potential in GYSELA normalized units
            - T0 (optional):  temperature at rpic in [keV]
        """
        if T0 is None:
            T0 = definitions.DENORMALIZATION['T'] # [keV] at rpic

        from scipy.interpolate import splev, splrep
        U = -T0 * 1e3 * Phi
        spl = splrep(r, U)
        return splev(r, spl, der=1)

    def get_Phi2D_stat():
        """ Return mean and variance of Phi(theta, r) along the last 1056
        diagnostics steps. They were extracted and stored in a file        previously. """
        [Phi2D_tavrg, Phi2D_tvar] = np.load(Path(definitions.DATA_DIR / 'diverse/Phi2D_tav.npy'))

        return Phi2D_tavrg, Phi2D_tvar

def get_experimental_prof(key='Ne', print_avail=False):

    p = Path(definitions.DATA_DIR) / '45511_prof.mat'
    mat = sio.loadmat(p)

    # print available profile keys:
    if print_avail:
        print(mat['prof'].dtype.names)

    prof = mat['prof'][key][0,0].flatten()
    return prof

def get_denormalized_nprof(choice='gys'):
    """
    Returns a profile (r/a, n) that has the same value
    at rpic (r/a=0.65) as the experimental profile.
    Args:
        - choice (str): 'gys' or 'exp'. The choice parameter handles
        how the profile is adapted in the region between rpic and r/a=1.1
        to fall off and reach zero at around r/a=1.2.
        'gys'--> profile as in GYSELA for r/a < 1.02 and as in experiment above
        'exp'--> profile as in GYSELA for r/a < rpic and as in experiment above
    """
    def smooth_transition(x, y1, y2, a, b):
        return y1 + (y2 - y1) * 0.5 * (1 + tanh((x - a) / b))

    # ion density profile (from last simulation frame)
    p_rprof = Path(definitions.DATA_DIR / 'ne_fluct/gysela/rprof/rprof_part_d03568.h5')
    f = h5py.File(p_rprof, 'r')
    n_gys = f['dens_FSavg'] # density n(r) averaged over flux surface
    r = np.linspace(1.27077224e-03, 1.3, 512) # gysela normalized radii

    rpic = 1.3 / 2
    ipic, _ = find_nearest(r, rpic)

    ne_exp = get_experimental_prof('Ne')
    r_exp  = get_experimental_prof('rho')
    ipic_exp, _ = find_nearest(r_exp, rpic)
    npic_exp = ne_exp[ipic_exp]

    # define transition point to zero
    rtr = 1.02
    itr, _ = find_nearest(r, rtr)
    ir = r > r[itr]
    n0_smooth = np.copy(n_gys)
    n0_smooth[ir] = smooth_transition(r[ir], 2*n_gys[itr], 0, r[itr], .045)

    if choice == 'gys':
        # de-normalization:
        n0 = n0_smooth / n0_smooth[ipic] * npic_exp

    elif choice == 'exp':
        ile, le = find_nearest(r_exp, 0.65)
        iri, ri = find_nearest(r_exp, 1.2)

        spline = interp1d(r_exp[ile:iri],
                            ne_exp[ile:iri]/npic_exp,
                            fill_value='extrapolate')
        ile, le = find_nearest(r, 0.67)
        iri, ri = find_nearest(r, 1.2)

        n0_smooth_2 = np.copy(n0_smooth/n0_smooth[ipic])
        n0_smooth_2[ile:iri] = spline(r[ile:iri])

        # de-normalization:
        n0 = n0_smooth_2 / n0_smooth_2[ipic] * npic_exp

    # reduce density w.r.t experiment to allow lower probing frequencies:
    n0 = n0 / 1.56086

    return r, n0


def synthetic_example():

    dx = 1.5e-4 # m
    nx = 1024
    ny = 1024 * 4
    lmin = 0.51e-2 # m
    lmax = 1.4e-2 # m
    beta = 20 * pi / 180 # rad

    map_args = (dx, nx, ny, lmin, lmax, beta)

    map = SynthTurbMap(map_args)
    map.show_spectrum()
    map.show_fluct()

    print(map.delta_ne.shape)

    plt.show()

def gysela_example():

    from fullwave2d.visualize import density_imstyle, get_default_imstyle
    Nx = 1000
    nx, ny = 700, 1100
    gamma = 0.42
    nxs = int(gamma * nx)
    nys = int(gamma * ny)
    ileft, ibottom = int(Nx * 0.7), int(Nx/2 - nys/2)

    slice_args = (nx, ny, gamma, ileft, ibottom)

    gysmap = GyselaMap.get_example_map(Nx=Nx)
    ne = gysmap.ne

    # ne_slice = get_slice(ne, *slice_args)
    #
    fig, ax = plt.subplots()
    imst = get_default_imstyle()
    # imst.imshow_kwargs['extent'] = gysmap.extent
    # im = imst.get_im(gysmap.dn_cart, ax)
    # im = density_imstyle.get_im(ne_slice, ax)

    density_imstyle.imshow_kwargs['extent'] = gysmap.extent
    im = density_imstyle.get_im(ne, ax)

    gysmap.show_limiter(ax)

    plt.show()


def _get_slice(turbmap, nx, ny, gamma, ileft, ibottom):
    """
    DEPRECATED: use get_slice() instead

    Take a slice of the turbulence map and change the
    resolution of the new slice by a factor of 1/gamma
    on each axis so as to match the shape of
    the computational domain. The returned array thus
    is a typically lower resolution version of turbmap
    with shape (ny, nx). The aspect ratio is left unchanged,
    so that the slice is fully defined by the the lower left
    corner (ileft, ibottom), the target size (ny, nx) and
    the scaling factor gamma.
    """

    # slice dimensions (ensuring the same aspect ratio)
    nxs = int(gamma * nx)
    nys = int(gamma * ny)

    # take a slice of shape (nxs,nys), starting from ileft and ibottom
    tslice = turbmap[ibottom:ibottom+nys, ileft:ileft+nxs]

    # change the resolution of the slice to match the
    # number of grid cells (ny,nx):
    spline = RectBivariateSpline(np.linspace(0,ny, nys),np.linspace(0,nx, nxs), tslice)

    new_grid_x = np.arange(nx)
    new_grid_y = np.arange(ny)

    new_slice = spline(new_grid_y, new_grid_x, grid=True)

    return new_slice

def load_gys_map(fname):
    """ DEPRECATED Loads a .h5 GYSELA turbulence map """
    f = h5py.File(fname, 'r')
    turbmap = f['fields']['map_dn']
    return np.array(turbmap)


if __name__ == '__main__':

    # synthetic_example()

    # gysela_example()

    # a = np.ones((300, 200))
    # b = tanh_edge_drop(a, None, 0.04, 0.01)
    # c = tanh_edge_drop(a, None, 0.01, 0.001)**3
    # d = b*c
    # # d = tanh_edge_drop(b, None, 0.02, 0.001)
    # fig, ax = plt.subplots()
    # ax.imshow(c)
    # fig, ax = plt.subplots()
    # ax.imshow(d)
    # print(b[0,0], b[0,100], b[-1, 100])
    # print(d[0,0], d[0,100], d[-1, 100])
    #
    # fig, ax = plt.subplots()
    # ax.plot(b[:,100])
    # ax.plot(d[:,100])
    # plt.show()




    plt.show()

#%%
import ctypes
from ctypes import POINTER
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from pathlib import Path
#%%
import pickle
import copy
from scipy.constants import c
import traceback
from fullwave2d import definitions
# from fullwave2d.main.misc import GaussianBeam, rotate2D
# # from fullwave2d.main.beamtracing import tracingdat
# from fullwave2d.main.turbulence_map import transform_resolution

from scipy.interpolate import RectBivariateSpline, interp1d

# Note: trying to load the data of an ongoing full-wave simulation
# leads to weird results and should be avoided

class InputData (object):
    """
    Data class defining the input parameters to full-wave simulations.
    The parameters can be saved using save_to_pickle()
    or loaded via load_pickle(). Use the print() method to
    get an overview of the relevant attributes of an instance.

    Args:
        name (string):  Simulation name. Files (input and output) are stored
                        at definitions.FW2D_DATA_DIR / name / or
                        definitions.FW2D_DATA_DIR / subdir / name / if subdir para-
                        meter is specified.
        f0 (float):     vacuum frequency of the probing beam
        nt (int):       number of time steps
        nx (int):       number of grid cells along x
        ny (int)        number of grid cells along y
        dx (float):     grid spacing (same is used for y)
        ne (ndarray):   density field (including fluctuations) of shape (ny, nx)

    Optional kwargs:
        mode (string):  wave polarisation, 'O' or 'X' for E||z or B||z, resp.
        b0 (ndarray):   magnetic field map (only relevant in X-mode) of shape
                        (ny, nx)
        extent (list):  [left, right, bottom ,top] extent in data coordinates
                        of ne(x,y) in a reference system of choice. The actual
                        size of the simulation box is larger because of the PML
                        and can be accessed via the full_extent variable
        npml (int):     number of cells for the PML absorbing layer
        reflmax (float):max reflexion coefficient for PML layer
        TFSF (int):     Total-Fields/Scattered-Fields interface position
        xante (int):    horizontal position (in cell units!) of the antenna
        angle (float):  absolute angle of incidence (in [deg]) of the beam
        angle_gola (float): relative angle of incidence (in [deg]) as defined
                        in the beam tracing code for Dreve and Difdop, respecti-
                        vely
        dtheta (float): Relative angle (in [deg]) between the (rotated) coordi-
                        nate axes x' and y' (in full-wave) with respect to the
                        actual horizontal and vertical axes x and y
        antenna (string): Name of the antenna modeled: None, 'difdop', or
                        'dreve'
        subdir (string): sub-directory: if specified (default is None), a sub-
                        directory will be created and files stored at definitions / FW2D_DATA_DIR / subdir / name /
        save_diag (bool): Whether to save diagnostics during the full-wave (de-
                        fault is True except for non-Root MPI processes)
    """
    def __init__(self, name, f0, nt, nx, ny, dx, ne,
                 **kwargs
                 ):
        
        self.machine = kwargs.pop('machine', 'irene')
        self.name = name
        self.f0   = f0
        self.lam  = c / f0
        self.nt   = nt
        self.nx   = nx
        self.ny   = ny
        self.dx   = dx
        # self.dny  = dny
        self.ne   = ne
        
        # default values if not specified by additional keyword arguments **kwargs:
        self.header     = kwargs.get('header', None)
        self.mode       = kwargs.get('mode', 'O')
        self.b0         = kwargs.get('b0', None)
        self.extent     = kwargs.get('extent', None)

        self.npml       = kwargs.get('npml', 8)
        self.reflmax    = kwargs.get('reflmax', 1e-4)
        self.TFSF       = kwargs.get('TFSF', self.npml + 10)
        self.full_extent= self.get_full_extent()
        # self.xante      = kwargs.get('xante', self.TFSF - 1)
        # self.yante      = kwargs.get('yante', 0) 
        #self.waist      = kwargs.get('waist', 30 * dx)
        # self.xante      = kwargs.get('xante', self.nx * self.dx)
        self.xante      = kwargs.get('xante', self.TFSF - 1)

        self.yante      = kwargs.get('yante', 0) - int(ny/2) * self.dx 
        self.waist      = kwargs.get('waist', 30 * dx) + (self.TFSF / 2 - 4) * self.dx

        self.angle      = -kwargs.get('angle', 10)
        self.angle_gola = kwargs.get('angle_gola', None)
        self.dtheta     = kwargs.get('dtheta', 0)
        self.antenna    = kwargs.get('antenna', None) # 'difdop' or 'dreve'
        self._antenna_setup()

        self.subdir     = kwargs.get('subdir', None)
        self.save_diag  =  kwargs.get('save_diag', True)

    def get_outp_dir(self, **kwargs):
        """
        Returns a path to this simulation's directory,
        which will be created or overridden once the simulation
        is run.
        """
        
        machine = kwargs.pop('machine', None)
        simdir = Path(definitions.FW2D_irene_DATA_DIR) if machine == 'irene' else Path(definitions.FW2D_marconi_DATA_DIR) if machine == 'marconi' else Path(definitions.FW2D_DATA_DIR)
        subdir = getattr(self, 'subdir', None)

        if subdir is not None:
            subdir = Path(simdir / subdir)
            if subdir.exists():
                pass
            else:
                try:
                    subdir.mkdir()
                except FileExistsError:
                    traceback.print_exc()
                    pass

            return Path(subdir / self.name)
        else:
            return Path(simdir / self.name)

    def save_to_pickle(self, path):
        """ Save this class instance to a binary .pkl file """
        with open(path / 'input.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load_pickle(name, subdir=None, **kwargs):
        """
        Returns an instance of this class which was saved previously to disk.
        """
                
        outp_dir = Path(definitions.get_sim_dir(name, subdir, **kwargs))
        with open(outp_dir / 'input.pkl', 'rb') as f:
            input_data = pickle.load(f)
            return input_data

    def print(self):
        """
        Display the class attributes
        """
        # dictionary of all the object's attributes:
        dict = self.__dict__
        form_spec = "{:<10} {:<10}"

        for key, val in zip(dict.keys(), dict.values()):

            # display only meaningful attributes
            if key not in ['ne', 'ampl_inc', 'phase_inc', 'b0']:
                if key=='f0':
                    print("{:<10} {:<10g}".format(key,val))
                    print("{:<10} {:<10g}".format('lambda',c/self.f0))
                    print(form_spec.format('n_crit', val**2 * 0.012404426))

                else:
                    print(form_spec.format(key,str(val)))


    def plot_ne(self, ax=None, imst=None):
        """ Display the density map """
        from fullwave2d.visualize import visualize

        if ax is None:
            fig, ax = plt.subplots()
        if imst is None:
            imst = copy.copy(visualize.density_imstyle)
            if self.extent is not None:
                imst.imshow_kwargs['extent'] = self.extent
        return imst.get_im(self.ne, ax)

    def get_xy_coords(self, full_ext=True, extent=None):

        ny, nx, dx = self.ny, self.nx, self.dx

        if full_ext:
            TFSF = self.TFSF
        else:
            TFSF = 0

        if extent is None:
            extent = self.extent

        if extent is None:
            x = np.linspace(-(nx+2*TFSF)*dx/2, (nx+2*TFSF)*dx/2, nx+2*TFSF)
            y = np.linspace(-(ny+2*TFSF)*dx/2, (ny+2*TFSF)*dx/2, ny+2*TFSF)
        else:
            l,r,b,t = extent
            x = np.linspace(l - TFSF*dx, r + TFSF*dx, nx + 2*TFSF)
            y = np.linspace(b - TFSF*dx, t + TFSF*dx, ny + 2*TFSF)
        return x, y

    def get_full_extent(self):
        """
        Return the simulation boundaries (left, right, bottom, top)
        in data coordinates (meters if dx is in meters) taking the
        PML layers into account.
        """
        if not hasattr(self, 'extent'):
            return None

        if self.extent is not None:
            (l,r,b,t) = self.extent
            dx, TFSF = self.dx, self.TFSF
            left = l - TFSF * dx
            right = r + TFSF * dx
            bottom = b - TFSF * dx
            top    =  t + TFSF * dx
            return np.array([left, right, bottom, top])
        else:
            return None

    def display_results(names, **kwargs):
        """ Convenience method to call visualize.display_results()
        by passing one or more simulation names. """
        import fullwave2d.visualize as visualize
        if type(names) != list and type(names) != np.ndarray:
            names = [names]
        subdir = kwargs.get('subdir', None)
        paths = [definitions.get_sim_dir(na, subdir) for na in names]
        inputs = [InputData.load_pickle(p) for p in paths]
        return visualize.display_results(inputs, **kwargs)

    def _antenna_setup(self):
        """
        Create the initial Gaussian field amplitude and phase at the antenna position (No wavefront curvature).
        """
        lam = self.lam
        w0 = self.waist
        dx = self.dx
        ny = self.ny
        TFSF = self.TFSF

        y_ant = self.yante - (TFSF)/2 * dx
        # y_ant = self.yante
        x,y = self.get_xy_coords(full_ext=True)

        # alp = np.deg2rad(self.angle)
        # dphase = - 2*pi * self.f0/c * dx * np.sin(alp)
        # ampl = np.flip(np.exp( - (np.cos(alp) * (y - y_ant) / w0)**2), axis=0)
        # phase = np.cumsum(np.ones_like(y) * dphase)
        
        # # Adjust phase to wrap around -PI to PI
        # phase = (phase + np.pi) % (2 * np.pi) - np.pi

        # self.ampl_inc = ampl
        # self.phase_inc = phase
        
        alp = np.deg2rad(self.angle)
        #dphase = - 2*pi * self.f0/c * dx * np.sin(alp)
        #ampl = np.flip(np.exp( - (np.cos(alp) * (y - y_ant) / w0)**2), axis=0)
        #phase = np.cumsum(np.ones_like(y) * dphase)
        
        aux = (np.cos(alp) * (y - y_ant) / (w0)) ** 2
        ampl = np.exp(-aux)
        ampl = np.flip(ampl, axis = 0)     
        
        dphase = - 2 * np.pi * self.f0 / c * dx * np.sin(alp)
        phase = np.cumsum(np.ones_like(y) * dphase)
        
        #if (phase.any() < -np.pi):
         #   phase += 2 * np.pi
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

        # Adjust phase to wrap around -PI to PI
        #phase = (phase + np.pi) % (2 * np.pi) - np.pi

        self.ampl_inc = ampl
        self.phase_inc = phase

    def antenna_setup(self):
        """
        EXPERIMENTAL function (not used yet):
        The idea is to setup amplitude and phase at the antenna position
        taking into account the wavefront curvature accumulated during vacuum-
        propagation.
        """
        lam = self.lam
        w0 = self.waist
        dx = self.dx
        ny = self.ny
        TFSF = self.TFSF
        alp = self.angle # incidence angle [deg] (wrt separatrix normal)

        a = 0.7 # minor radius [m]
        R = 2.4 # major radius   [m]
        rgola = 4.3 # radial distance antenna-plasma core  [m]
        d_ant = rgola - R - 1.2 # radial distance antenna-separatrix [m]

        self.yante = self.yante - (TFSF) * dx
        y_ant = self.yante 
        y = np.linspace(-(ny+TFSF)*dx/2, (ny+TFSF)*dx/2, ny+2*TFSF)
        x = np.array([0])
        X, Y = np.meshgrid(x,y)

        origin = [0, 0] # rotate about this point
        Xr, Yr = rotate2D(X, Y, alp, origin)

        gauss_beam = GaussianBeam(lam, w0)
        ampl = gauss_beam.ampl(Xr - d_ant, Yr)
        ampl[:TFSF] = 0.0
        ampl[-TFSF:] = 0.0
        phase = gauss_beam.phase(Xr - d_ant, Yr)

        self.ampl_inc = ampl.flatten() / np.max(ampl)
        self.phase_inc = phase.flatten()

    def get_tracing_data(self, antenna='difdop'):
        # """ Convenience function for calling tracingdat.get_tracing_data()."""
        # if self.angle_gola is None:
        #     print('No attribute "angle_gola" attached to this input data yet.')
        #     return None
        # else:
        #     f0  = np.around(self.f0 / 1e9, 1)
        #     ang = np.around(self.angle_gola, 1)
        #     if antenna=='difdop':
        #         tracdat = tracingdat.difdop_data.get_tracing_data(f0, ang)
        #     elif antenna=='dreve':
        #         tracdat = tracingdat.dreve_data.get_tracing_data(f0, ang)
        #     else:
        #         tracdat = None

        #     if tracdat is None:
        #         print('No beam tracing data available for this pair of frequency and poloidal angle')
        #     return tracdat
        pass # TODO: to be put elsewhere (not in wrapper.py)

    def delete_simulation(names, subdir=None):
        """ Delete all simulation files and the corresponding directory."""

        if (type(names)!=list and type(names)!=np.ndarray):
            names = [names]
        for name in names:
            simdir = definitions.get_sim_dir(name, subdir)
            if simdir.exists():
                # clear all files in that directory
                [Path.unlink(p) for p in simdir.iterdir()]
                Path.rmdir(simdir)
                print('succesfully removed simulation \'{}\' '.format(simdir))
            else:
                print('simulation \'{}\' does not exist'.format(simdir))


class OutputData (object):

    """
    Handles full-wave simulation oputput data.
    Converts .dat files (output of the C code) into .npy files and allows
    compressing files if necessary.

    Args:
    - name (string): Directory name containing the data files. The files are
                     assumed to be located at definitions.FW2D_DATA_DIR / name / or
                     definitions.FW2D_DATA_DIR / subdir / name / if subdir is
                     specified
    - subdir (string or None, optional): sub-directory, defaults to None
    - compress_factor (int, optional): The resolution of the field can be
                     reduced by this factor (along x and y axes).

    usage:
    outp = OutputData(name, subdir)

    The following attributes can be accessed (if the corresponding file exists):
        - outp.ampl: amplitude at the antenna (during full-wave sim.)
        - outp.phase: phase    at the antenna (during full-wave sim.)
        - outp.ez: z-comp. of the E (or B) field at the last full-wave sim. step
        - outp.ez_inc: reference field (without plasma i.e. free propagation)
        - outp.doppler_data: Amplitude and phase evolution (as a result of
                            parallelized simulation sequence)
    """
    def __init__(self, name, subdir=None, compress_factor=1, **kwargs):

        self.name = name
        self.subdir = subdir
        self.outp_dir = Path(definitions.get_sim_dir(name, subdir, **kwargs))

        if Path(self.outp_dir / 'ant_signal_t.npy').exists():
            self.ampl, self.phase = OutputData.read_antenna_data(self.outp_dir / 'ant_signal_t.npy')

        # the perp. field component (electric for O-mode, magnetic for X-mode):
        ez = np.load(self.outp_dir / 'ez_t.npy')
        self.ez = OutputData.compress_data(ez, n=compress_factor)

        if Path(self.outp_dir / 'ez_t_inc.npy').exists():
            ez_inc = np.load(self.outp_dir / 'ez_t_inc.npy')
            self.ez_inc = OutputData.compress_data(ez_inc, n=compress_factor)

        # if it exsists, load the Doppler antenna data:
        if Path(self.outp_dir / 'ampl_phase.npy').exists():
            self.doppler_data = np.load(self.outp_dir / 'ampl_phase.npy')

    def save_to_pickle(self, path, mkdir=False):
        """ Save this class instance to a binary .pkl file """
        if mkdir:
            p = Path(path)
            if not p.exists():
                p.mkdir(parents=True)

        with open(path / 'output.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load_pickle(name, subdir=None):
        """
        Returns an instance of this class which was saved previously to disk.
        """
        outp_dir = Path(definitions.get_sim_dir(name, subdir))
        with open(outp_dir / 'output.pkl', 'rb') as f:
            output_data = pickle.load(f)
            return output_data

    def txt_to_npy(fname, override=True):
        """
        Checks if "fname" is a binary ".npy" array.
        If not, it is converted and saved as such.
        This will save a lot of time the next time
        one needs to access the same data.
        The old text file is overridden if not specified
        otherwise.
        """
        if fname.suffix != '.npy':
            pf = Path(fname)
            fname_np = pf.parent / (pf.stem + '.npy')
            if Path(fname_np).exists():
                fname = fname_np
            else:
                data = np.loadtxt(fname)
                np.save(fname_np, data)
                if override:
                    # remove the deprecated txt/dat file
                    Path(fname).unlink()
                fname = fname_np

        return fname

    def read_antenna_data(fname, override=True):
        """
        Loads the recieved signal amplitude and phase
        time traces, assuming left and right columns
        are amplitude and phase, respectively.
        """
        # assert we are loading a numpy a file,
        # if not, a .npy is created:
        fname = OutputData.txt_to_npy(fname, override=override)
        data = np.load(fname).reshape(-1, 2)
        ampl, phase = data[:,0], data[:,1]
        return  ampl, phase

    def read_2d_data(fname, nx, ny, override=True):
        """
        Loads the two dimensional (spatial) field
        data stored in file location "fname". It is assumed that
        the data has the form (nt*nx, ny). It is reshaped
        into an animation compatible form where nt is
        inferred automatically, given nx and ny.
        Usually, the field is stored at the last time step only, so
        that nt=1.
        """
        # assert we are loading a numpy a file,
        # if not, a .npy is created:
        fname = OutputData.txt_to_npy(fname, override=override)
        data = np.load(fname).reshape(-1, ny, nx)
        nt = data.shape[0]
        if nt > 1:
            np.save(fname, data)
        print("Read file containing {}".format(nt) + \
        " frames of shape ({}, {}).".format(ny, nx))

        return data

    def compress_data(data, n=4):
        """
        Reduce resolution along x and y axes by cubic interpolation.
        Args:
        - data (ndarray): 2 or 3-dimensional array. The data is compressed along
                          the last two axes.
        - n (float):      compress by this factor
        """
        if data.ndim == 3:
            (nt, ny, nx) = data.shape
            ny_new, nx_new = int(ny/n), int(nx/n)
            data_new = np.zeros((nt, ny_new, nx_new))
            for t in range(nt):
                data_new[t] = transform_resolution(data[t], ny_new, nx_new)

        elif data.ndim == 2:
            (ny, nx) = data.shape
            ny_new, nx_new = int(ny/n), int(nx/n)
            data_new = transform_resolution(data, ny_new, nx_new)
            
        else:
            raise ValueError('data must be 2 or 3-dimensional, but has shape {}'.format(data.shape))

        return data_new

    def extract(name, subdir, new_subdir, nez=4, nne=6):
        """
        Reads simulation input and output data, and compresses the field
        data and the input density along each spatial axis by a factor
        of nez and nne, respectively. The compressed data is saved
        as input.pkl and output.pkl in a new sub-directory.
        """
        outp_dir = definitions.get_sim_dir(name, subdir)

        inp = InputData.load_pickle(name, subdir)

        ne = OutputData.compress_data(inp.ne, nne)
        inp.ne = ne

        outp = OutputData(name, subdir=subdir, compress_factor=nez)
        outp.save_to_pickle(definitions.FW2D_DATA_DIR / new_subdir / name, mkdir=True)
        inp.save_to_pickle(definitions.FW2D_DATA_DIR / new_subdir / name)
        
    def display_results(self, **kwargs):
        """ Convenience method to call visualize.display_results()
        by passing one or more simulation names. """
        import fullwave2d.visualize as visualize
        inp = InputData.load_pickle(self.name, self.subdir)
        return visualize.display_results(inp, **kwargs)


### Python-C interfacing ###

# load the shared object compiled from the C file
# to which we are creating a python interface
libname = Path(definitions.SRC_DIR).joinpath('py_maxwell_interface.so')
maxw_lib = ctypes.cdll.LoadLibrary(libname)

# this type declaration allows passing 2d numpy arrays from python
# to double pointers in c:
doublepp = np.ctypeslib.ndpointer(dtype=np.uintp)


maxw = maxw_lib.main
maxw.restype = None
# declare argument types expected by the C function:
maxw.argtypes = [
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_double,
                    ctypes.c_int,       # npml
                    ctypes.c_double,    # reflmax
                    ctypes.c_int,       # TFSF
                    ctypes.c_int,       # xante

                    ctypes.c_char,
                    doublepp,           # ne
                    doublepp,           # b0
                    POINTER(ctypes.c_double),
                    POINTER(ctypes.c_double),
                    doublepp,
                    doublepp,

                    ctypes.c_bool,
                    ctypes.c_char_p
                ]

def to_double_pointer(arr):
    """
    Construct the (double **) version of a numpy.ndarray for compatibility
    with the ctypes interface from python to C.
    """
    return (arr.ctypes.data + np.arange(arr.shape[0]) * arr.strides[0]).astype(np.uintp)


def fw2d_wrapper(inp, make_outp_dir=True):

    """
    Performs the type conversion via ctypes in order to pass
    the input data to the Maxwell solver in C.
    Note: Type conversion can be very tricky and lead to
          segmentation fault in C if not done properly.

    Args:
        inp (wrapper.InputData): instance of the input class
        make_outp_dir (bool):   Whether to create a directory to store the
                                output data (default is True).
    Returns:
        ampl  (double): signal amplitude at last simulation step
        phase (double): signal phase at last simulation step

    # TODO: implement subprocess option
    """
    
    if make_outp_dir and inp.save_diag:

        # Create a directory in which to store the output.
        # If that directory already exists, it will be overridden!
        outp_dir = inp.get_outp_dir()
        if outp_dir.exists():
            # clear old files
            [Path.unlink(p) for p in outp_dir.iterdir()]
        else:
            outp_dir.mkdir()

        # save the input parameters to binary, so that they can
        # be accessed and/or reused (see InputData.load_pickle()):
        inp.save_to_pickle(outp_dir)

    else:
        outp_dir = 'None'


    # bring the array elements into the right order (flip along x and y)
    ne = np.flip(inp.ne, axis=0)
    ne = np.flip(ne, axis=1)
    # # The following line seems redundant but for some strange reason is absolutely necessary! :
    ne = np.array(ne)
    # ne = np.array(inp.ne, dtype=np.double)  # preserve the correct orientation

    nepp = to_double_pointer(ne)

    if inp.mode == 'O':
        b0 = np.zeros((inp.ny, inp.nx))
    else:
        if inp.b0 is None:
            print('Warning: X-mode but no magnetic field provided. Assuming B=0.')
            b0 = np.zeros((inp.ny, inp.nx))
        else:
            b0 = inp.b0

    b0pp = to_double_pointer(b0)

    # amplitude and phase at the antenna, to be passed by reference
    ampl = ctypes.c_double(0.0)
    phase = ctypes.c_double(0.0)

    ampl_incpp = to_double_pointer(inp.ampl_inc)
    phase_incpp = to_double_pointer(inp.phase_inc)

    maxw(
         ctypes.c_double(inp.f0),
         ctypes.c_int(inp.nt),
         ctypes.c_int(inp.nx),
         ctypes.c_int(inp.ny),
         ctypes.c_double(inp.dx),
         ctypes.c_int(inp.npml),
         ctypes.c_double(inp.reflmax),
         ctypes.c_int(inp.TFSF),
         ctypes.c_int(int(inp.xante)),

         ctypes.c_char(str(inp.mode).encode('utf-8')),

         nepp,
         b0pp,
         ctypes.byref(ampl),
         ctypes.byref(phase),

         ampl_incpp,
         phase_incpp,
         ctypes.c_bool(inp.save_diag),
         ctypes.c_char_p(str(outp_dir).encode('utf-8'))
        )

    # automatically convert the .dat files into .npy files for efficiency
    fn_field     = Path(outp_dir) / 'ez_t.dat'
    fn_field_inc = Path(outp_dir) / 'ez_t_inc.dat'
    # fn_field_anim= Path(outp_dir) / 'ez_anim.dat'
    fn_antenna   = Path(outp_dir) / 'ant_signal_t.dat'

    for fn in [fn_field, fn_field_inc]:
        Nx = (inp.nx+2*inp.TFSF)
        Ny = (inp.ny+2*inp.TFSF)
        if fn.exists():
            # if fn==fn_field_anim:
            #     Nx = (Nx) // 5
            #     Ny = (Ny) // 5
            OutputData.read_2d_data(fn, Nx, Ny, override=True)

    if fn_antenna.exists():
        OutputData.read_antenna_data(fn_antenna, override=True)
        
    
    # if make_outp_dir: # exctract data and compress it
    #     subdir = inp.subdir
    #     if subdir is None:
    #         new_subdir = inp.name + '_compressed'
    #     else:
    #         new_subdir = subdir + '_compressed'

    #     OutputData.extract(inp.name, subdir, new_subdir)


    return ampl.value, phase.value

# %%

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
# %%

# %%
# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import time

from fullwave2d.main.turbulence_map import lin_prof
from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData


simname='airy_reflection'

def _run_simulation():
    """
    Create a linear density profile and perform a fullwave calculation simulating a reflection in O-mode to check with the analytical solution (airy function) for the resulting standing wave.
    """

    # global simulation parameters
    f0 = 55e9 # probing frequency [Hz]
    nt = 3500 # number of time steps
    nx = int(400)
    ny = int(700)
    dx = 2.75e-4 # spatial resolution [m]
    waist = 100 * dx
    yante = 0
    angle = 0


    # define a place holder for the density map:
    ne = np.zeros(nx * ny).reshape((ny, nx)).astype(np.double)
    # NOTE: The axes (x,y) of ne are reversed, because the source script
    # maxwell_2d_omode.c expects a 2d array of shape (ny, nx)

    # intitalize with linear background profile and
    # cutoff at halfway in the plasma
    lin_prof(ne, f0, cut=.7, start=0)
    ne = np.flip(ne, axis=1)
    # NOTE: check plasma boundary position with respect
    # to position of antenna when it comes to realistic simulations


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

    InputData.display_results([simname]);

def _display_intensity_pattern():

    inp = InputData.load_pickle(simname)
    out = OutputData(simname)

    _ny, _nx = out.ez.shape # note _ny and _nx are the dimensions of the output data, which are larger than the simulation domain (due to the PMLs)
    ez_radial_prof = out.ez[_ny//2,:]
    I_radial_prof = ez_radial_prof**2 / np.max(ez_radial_prof**2)

    x = (np.arange(_nx) - (_nx - inp.nx)/2 ) * inp.dx # correct for the extra PMLs to have x=0 at the position of the wave source
    fig,ax = plt.subplots()
    ax.plot(x, I_radial_prof, label='full-wave')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('$|E_z|^2$ [a. u.]')

def _display_airy_prediction():


    from scipy.constants import c
    from scipy import special

    def airy_lin_prof(x, f0, Ln):
        """
        Ln (turning point)
        """
        om = f0 * 2 * pi
        eta = (om**2 / c**2 / Ln)**(1/3) * (x - Ln)
        
        ai, aip, bi, bip = special.airy(eta)
        E0 = 1.0
        u = om * Ln / c
        E = 2 * E0 * pi**0.5 * u**(1/6) * ai
        phase = (2/3 * u - pi/4)
        E *= np.cos(phase)
        
        #return eta, E
        return eta, ai

    inp = InputData.load_pickle(simname)
    x = np.arange(inp.nx) *  inp.dx
    Ln = 0.7*inp.nx* inp.dx
    eta, E = airy_lin_prof(x, inp.f0, Ln)


    ax = plt.gca()
    ax.plot(x, E**2 / np.max(E**2), ls='--', label='analytical $A_i(\eta)$')
    ax2 = ax.twiny()
    ax2.set_xlim(eta[0], eta[-1])
    ax2.set_xlabel('$\eta = (\omega^2/c^2 L_n)^{1/3}\, (x - L_n)$')
    ax.axvline(Ln, color='black',ls='--', label='cut-off $L_n$')
    ax.set_xlim(x[0], x[-1])
    ax.legend()


if __name__ == '__main__':
    pass

    _run_simulation()

    #%%

    _display_intensity_pattern()
    #%%
    _display_airy_prediction()

    plt.show()
# %%


#%%
from pathlib import Path

def get_simname(f0, angle, mode, simulation, **kwargs):
    if simulation == 'spectrum_study':
        extra_name  = 'spect'
        extra_value = kwargs.pop('spect', 0)
    elif simulation == 'tilt_angle':
        extra_name  = 'beta'
        extra_value = kwargs.pop('beta', 70)
        
    return simname.format(int(f0*1e-9), abs(angle), mode, extra_name, extra_value)


#general name 
#simname = 'spect{}_f{}_angle{}_{}'
simname = 'f{}_angle{}_{}_{}{}'
# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# source scripts directory
SRC_DIR = ROOT_DIR.joinpath('src/')

# default Data directory, change it with
DATA_DIR = ROOT_DIR.parent.joinpath('data/')

# fw2d Simulation output directory
FW2D_DATA_DIR = DATA_DIR.joinpath('fw2d/')
FW2D_irene_DATA_DIR = DATA_DIR.joinpath('fw2d_irene/')
FW2D_marconi_DATA_DIR = DATA_DIR.joinpath('fw2d_marconi/')

# fw2d analysed results output directory 
DATA_ANALYSED_DIR        = ROOT_DIR.parent.joinpath('data_analysed/')
DATA_ANALYSED_maroni_DIR = DATA_ANALYSED_DIR.joinpath('marconi/')
DATA_ANALYSED_irene_DIR  = DATA_ANALYSED_DIR.joinpath('irene/')

# fw2d analysed figures output directory 
FIG_ANALYSED_DIR        = ROOT_DIR.parent.joinpath('fig_analysed/')
FIG_ANALYSED_maroni_DIR = FIG_ANALYSED_DIR.joinpath('marconi/')
FIG_ANALYSED_irene_DIR  = FIG_ANALYSED_DIR.joinpath('irene/')

# beam tracing directory
BEAM_DIR = ROOT_DIR.joinpath('main/beamtracing/')

# GYSELA (mainly Phi2D, rprof and init_state .h5 files) directory
GYS_DIR = DATA_DIR.joinpath('ne_fluct/gysela/')

# WEST experimental data 
WEST_DIR = ROOT_DIR.parent.joinpath('WEST_data/')
DENSITY_DIR = WEST_DIR.joinpath('density/')
EQUI_DIR = WEST_DIR.joinpath('equilibrium/')


def get_sim_dir(sim_name, subdir=None, **kwargs):
    """ Returns the absolute pathname to the
    simulation directory with name sim_name"""
    
    machine = kwargs.pop('machine', None)
    FW2D_DATA = FW2D_irene_DATA_DIR if machine == 'irene' else FW2D_marconi_DATA_DIR if machine == 'marconi' else FW2D_DATA_DIR
    
    if subdir is None:
        return Path(FW2D_DATA) / sim_name
    else:
        return Path(FW2D_DATA) /subdir / sim_name

def get_analysed_path(sim_name, subdir=None, **kwargs):
    """ Returns the absolute pathname to the
    data analysed with name sim_name"""
    
    machine = kwargs.pop('machine', 'irene')
    FW2D_DATA = DATA_ANALYSED_irene_DIR if machine == 'irene' else DATA_ANALYSED_maroni_DIR
    
    sim_name += '_' + kwargs.pop('fit', 'FFT')
    if subdir is None:
        return (Path(FW2D_DATA) / sim_name).with_suffix('.mat')
    else:
        return (Path(FW2D_DATA) /subdir / sim_name ).with_suffix('.mat')

def get_figures_path(sim_name, subdir=None, **kwargs):
    """ Returns the absolute pathname to the
    data analysed with name sim_name"""
    
    machine = kwargs.pop('machine', 'irene')
    FW2D_DATA = FIG_ANALYSED_irene_DIR if machine == 'irene' else FIG_ANALYSED_maroni_DIR
    figure = kwargs.pop('fig_type', 'complete')
    
    if subdir is None:
        return (Path(FW2D_DATA) / sim_name / figure).with_suffix('.pdf')
    else:
        return (Path(FW2D_DATA) /subdir / sim_name / figure).with_suffix('.pdf')

from scipy.constants import e, m_p
from numpy import sqrt

def get_denorm_B0(T0, rhostar=1/250, Z0=1, A0=2, a=0.7):
    """
    Get denormalized magnetic on-axis field in [T] for a given temperature
    in [keV] at rpic/a=0.65 .
    """
    B0 = rhostar * Z0 * e / sqrt(A0 * m_p * T0 * 1e3 * e) / a
    return B0

def get_fci(B0, Z0=1, A0=2):
    """
    Get ion cyclotron frequency in [rad/s] for an (on-axis) field B0 in [T].
    """
    return Z0 * e * B0 / A0 / m_p

def get_vth(T0, A0=2):
    """ Reference thermal velocity in [m/s] for a reference
     temperature T0 in [keV] """
    VT0 = sqrt(T0 * 1e3 * e / A0 / m_p)
    return VT0


DENORMALIZATION = {
                    'B0'    :   1.666,         # on axis magnetic field [T]
                    # 'R0'    :   2.4,         # major radius [m]
                    # 'eps'   :   1./3.2,       # inverse aspect ratio
                    'fci'   :   79.82e6,     # ion cyclotron freq. [Hz]
                    'vth'   :   164208.,    # thermal velocity [m/s]
                    'n'     :   3.8766e19,    # density at r_pic [m⁻³]
                    'T'     :   5.63e-1,    # temperature at r_pic [keV]
                    }


### DEPRECATED ###

# Synthetic fuctuation maps directory
SYNTH_MAP_DIR = DATA_DIR.joinpath('ne_fluct/synthetic_map/')

# %%

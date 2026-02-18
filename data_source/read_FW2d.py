#%%
from data_source.sigprocessing import init_specobjs, perform_specobj_fits
from data_source.fit_utils import gaussian, lorentzian, taylor
import numpy as np 
import matplotlib.pyplot as plt 
from matlabtools import Struct
from scipy import constants as cnst

def _get_fDop_from_fit_results(fit_params, xscale=1e3, f_cog=None):
    """Returns a dictionary containing the estimated Doppler frequency and errors in Hz, using xscale as theconversion factor between the normalized fit range and the frequency.
    Args:
    -----
        fit_params: dict
            Dictionary containing the fit params of the different curves.
        xscale: float
            Scaling factor for the frequency.
    """
    
    fDop_fits = np.array([params[1] for params in fit_params.values()]) * xscale
    
    if f_cog is not None:
        fDop_fits = np.append(fDop_fits, f_cog)
    
    try:
        if len(fDop_fits) <= 1 or np.isnan(fDop_fits).sum() >= len(fDop_fits)-1: # avoid underestimating the spread if we only have one valid estimation
            fDop_spread = np.nan
        else:
            fDop_spread = np.nanmax(fDop_fits) - np.nanmin(fDop_fits)
            
        fDop_median = np.nanmedian(fDop_fits)
    except ValueError:
        fDop_spread = np.nan
        fDop_median = np.nan
        
    if fDop_median > 0:
        fDop_max = np.nanmax(fDop_fits)
        fDop_min = np.nanmin([np.nanmin(fDop_fits), fDop_median - 0.5 * (fDop_max - fDop_median)])
    else:
        fDop_min = np.nanmin(fDop_fits)
        fDop_max = np.nanmax([np.nanmax(fDop_fits), fDop_median + 0.5 * (fDop_median - fDop_min)])
        
    
    outp = Struct(
        fDop = fDop_median,
        fDop_max = fDop_max,
        fDop_min = fDop_min,
        dfDop = fDop_max - fDop_min,
    )
    for curve_type, params in fit_params.items():
        outp[f'fDop_{curve_type}'] = params[1] * xscale
    
    
    return outp 
    
class FW2DReader(Struct):

    def __init__(self, subdir, **kwargs):
        
        self.subdir = subdir
        
    
    
    @classmethod
    def compute_profile(cls, subdir, machine = 'irene', **kwargs):
        
        specobjs = init_specobjs(subdir, machine = machine)

        outp = Struct()
        angle    = []
        freqGHz  = []
        fDop     = []
        fDop_min = []
        fDop_max = []


        for s in specobjs:
            perform_specobj_fits(s)
            _out = _get_fDop_from_fit_results(s.fit_params, s.xscale)
            angle.append(s.header.angle)
            freqGHz.append(s.header.f0)
            fDop.append(_out.fDop)
            fDop_min.append(_out.fDop_min)
            fDop_max.append(_out.fDop_max)

        outp.angle    = np.array(angle)
        outp.freqGHz  = np.array(freqGHz)
        outp.fDop     = np.array(fDop)
        outp.fDop_min = np.array(fDop_min)
        outp.fDop_max = np.array(fDop_max)
        outp.k_perp   = 4 * np.pi * outp.freqGHz * 1e9 / cnst.c * np.sin(outp.angle * np.pi / 180) / 100 #cm^-1
        outp.v_perp   = (2 * np.pi * outp.fDop) / (outp.k_perp * 1e2)
        outp.v_min    = (2 * np.pi * outp.fDop_min) / (outp.k_perp * 1e2)
        outp.v_max    = (2 * np.pi * outp.fDop_max) / (outp.k_perp * 1e2)

        outp.specobjs = specobjs
        
        return outp
    

def plot_semilog10(xdata, ydata, ax = None, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize = (5, 4))
    ax.plot(xdata, 10 * np.log10(ydata), **plot_kwargs)
    
def FW2DSpectraVisualize(s, ax = None, show_curves=['rawPSD', 'fittedPSD', 'oddPSD', 'fit_odd', 'fit_gaussian', 'fit_lorentzian', 'fit_taylor', 'annotations'], recenter = False, **kwargs):
    
    color_dict = {'raw': 'r','gaussian': '#000090', 'lorentzian': '#00A050','taylor': 'black','odd': '#B40090','even': 'lightgreen'}
    label_dict = {'raw': 'rawPSD','gaussian': 'Gaussian fit','lorentzian': 'Lorentzian fit','taylor': 'Taylor fit',}
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (5, 4))
        
    fDop_fits = np.array([params[1] for params in s.fit_params.values()]) * s.xscale
    fDop_median = np.nanmedian(fDop_fits)

    fD = fDop_median if recenter else 0.0
    print(fD)
    for show_c in show_curves:
        xfit = np.linspace(np.min(s.f), np.max(s.f), 1000) / s.xscale
        if show_c == 'rawPSD':
            plot_semilog10((s.f - fD) / s.xscale, s.P + s.P_noise, ax = ax, c = color_dict['raw'], label = label_dict['raw'])
        # elif show_c in ['fit_gaussian', 'fit_lorentzian', 'fit_taylor']:
        #     if show_c == 'fit_gaussian':
        #         curve_type = show_c.split('_')[1]
        #         yfit = gaussian(xfit , *s.fit_params.gaussian) * s.yscale + s.P_noise
        #     elif show_c == 'fit_lorentzian':
        #         curve_type = show_c.split('_')[1]
        #         yfit = lorentzian(xfit , *s.fit_params.lorentzian) * s.yscale + s.P_noise
        #     elif show_c == 'fit_taylor':
        #         curve_type = show_c.split('_')[1]
        #         yfit = taylor(xfit , *s.fit_params.taylor, dt = s.dt * s.xscale) * s.yscale + s.P_noise
        #     plot_semilog10(xfit - fD / s.xscale , yfit, ax = ax, c = color_dict[curve_type], label = label_dict[curve_type])
    
    if not recenter:
        ax.axvline(fDop_median / s.xscale, c = 'b', label = r'$f_{D}$ = %.3f kHz' %(fDop_median / s.xscale))
    else:
        ax.set_xlim(-1, 1)
    ax.grid(c = 'silver', ls = '--', lw = 0.5)
    ax.set_xlabel('f [kHz]', fontsize= 10)
    ax.set_ylabel('PSD [a.u.]', fontsize = 10)
    ax.legend()
    ax.text(x=1, y=0.15, s=r'$f_{prob}$ = %d GHz -- $\theta$ = %d° -- %s mode' %(s.header.f0, s.header.angle, s.header.mode),
        color = '#323232' ,
        rotation=270, 
        fontsize=10, 
        transform=plt.gca().transAxes)
    ax.set_title( r'$f_{D}$ = %.3f kHz' %(fDop_median / s.xscale))
# %%

if __name__ == '__main__':
    subdir = 'mixed_advection_2'
    
    out = FW2DReader.compute_profile(subdir)
    
#%%
    i = 0  
    s = out.specobjs[i]
    
    # %matplotlib widget
    FW2DSpectraVisualize(s, recenter = False)
    

# %%
    out.fDop[i] = -724

# %%

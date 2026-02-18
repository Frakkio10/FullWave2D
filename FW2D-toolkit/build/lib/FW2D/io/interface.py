#%%
import numpy as np 
import warnings
from pathlib import Path
from matlabtools import Struct 

from FW2D.io.utils import input_prompt
from fullwave2d.config import definitions as defs
from fullwave2d.core.wrapper import InputData, OutputData

def get_subfolders(folder_path):
    return [f for f in folder_path.iterdir() if f.is_dir()]

class DataInterface(Struct):
    
    def __init__(self, subdir, machine = 'irene', verbose = False):
        
        self.subdir  = subdir 
        self.machine = machine
        
        
        pathdir, params = self.get_FW_params()
        self.pathdir = pathdir
        self.params  = params
        
    def get_FW_params(self,):
        DATA_DIR = Path(defs.FW2D_irene_DATA_DIR) if self.machine == 'irene' else Path(defs.FW2D_marconi_DATA_DIR) if self.machine == 'marconi' else Path(defs.FW2D_DATA_DIR)    
        pathdir = DATA_DIR.joinpath(self.subdir)
        
        folders = get_subfolders(pathdir)
        
        F     = []
        theta = []
        waist = []
        mode  = []
        name  = []
        
        for simname in folders:
            inp = InputData.load_pickle(simname, subdir = self.subdir, machine = self.machine)
            F.append(int(inp.f0 * 1e-9))
            theta.append(-inp.angle)
            waist.append(int(inp.waist / inp.dx -  int((inp.TFSF / 2 - 4))))
            mode.append(inp.mode)
            name.append(inp.name)
        
        params = Struct()
        params.F     = np.array(F)
        params.theta = np.array(theta)
        params.waist = np.array(waist)
        params.mode  = np.array(mode)
        params.name  = np.array(name)
        params.Nbsim = params.F.size
        params.isims = np.arange(params.Nbsim) + 1
        
        return pathdir, params
    
    def get_signal(self, isim = None):
        
        if isim is None:
            isim = input_prompt(f'Which simulation? from 1 to {self.params.Nbsim +1}')
        
        out = OutputData(self.params.name[isim -1], subdir=self.subdir, machine = self.machine)
        #amplitude and phase of the field from the doppler data of the FW simulations
        x, y = out.doppler_data[:, 0], out.doppler_data[:, 1]
        t = np.linspace(0, 1e-4 * x.size, x.size)
        return t, x, y
    
    def _get_sim_choice(self, freq_choice):
        """Convenience method to call get_sim_choice()"""
        return DataInterface.get_sim_choice(self.params, freq_choice)

    @staticmethod
    def get_sim_choice(params, sim_choice):

        par = params # shortcut
        sim_avail = np.arange(1, par.Nbsim+1)

        try:

            if type(sim_choice) is np.ndarray:
                pass

            elif sim_choice == 'all':
                sim_choice = sim_avail

            elif type(sim_choice) is int:
                sim_choice = [sim_choice]
            
            elif sim_choice is None or sim_choice == []:
                default = 'all'  # sim_avail

                sim_choice = input_prompt(f'Which simulation choice (array, int, or "all") ? [1, 2, 3, ... , {par.Nbsim}] Defaults to all.', default=default, assert_type=(
                    tuple, list, int, str, np.ndarray, range))

                if type(sim_choice) is int:
                    sim_choice = [sim_choice]
                elif type(sim_choice) is str:
                    if sim_choice == 'all':
                        # sim_choice = sim_avail
                        return DataInterface.get_sim_choice(params, sim_choice)
                    else:
                        raise AssertionError(
                            f'Unknown string {sim_choice} input as <sim_choice>.')

                sim_choice = np.array(sim_choice)
            else:
                sim_choice = np.array(sim_choice)
                
            # careful with indexing, frequency choice starts at 1:
            if sim_choice[0] == 0:
                sim_choice += 1
                print(
                    f'Frequency choice starts at 1 (not 0), so array was shifted by +1.')

            # is the sim_choice contained in sim_avail:
            for f in sim_choice:
                assert f in sim_avail, f'Frequency choice {f} is not available. Available choices are {sim_avail}.'

        except AssertionError as e:
            print(e)
            return DataInterface.get_sim_choice(params, None)

        return np.array(sim_choice)
# %%

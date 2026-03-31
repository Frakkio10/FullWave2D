

#%% 
"""Example calls:
    
    AUG:
    - DBSgui aug 42593 --modules fit1 fit2 fit3 fit4 fit5 fit6 fit7 --comb --isweep 49 -ch [1,2,3,4,5,6,7] # loads the fitspec modules for each of the 7 W-band comb channels
"""

from pyqtplotlib.windows import TabsWindow


class MainWindow(TabsWindow):
    def __init__(self, module_widgets, module_titles=None, parent=None):
        super().__init__(module_widgets, module_titles, parent)
        
        # Set up the main window
        self.setWindowTitle("FW2Dgui")
        self.setGeometry(100, 100, 1200, 900)
        
        self.modules = {} # dictionary for easier referencing to the active modules
        for name, module in zip(module_titles, module_widgets):
            module.parent = self # assign this to be the parent of each module
            self.modules[name] = module # this way, the module can reference the main window, and other modules

        

def main():
    import sys
    import numpy as np
    import argparse
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    
    # pass arguments from the command line
    parser = argparse.ArgumentParser(
        description='FW2Dgui: A General command line interface to load GUI modules for analysis of FW2D data.')

    parser.add_argument('subdir', type=str,
                        help='insert the subdir you want to analyse')
    parser.add_argument('--isims', '--isim', default='all', 
                        help='The simulation indices (starting from one) to treat. Can be an int, list, tuple, [], or "all"')
    parser.add_argument('--modules', nargs='*', default=['fit'],        
                        help='modules to load among: "raw", "spec", "fit1", "fit2"')
    parser.add_argument('--machine', default='irene', type=str,
                        help='machine name ("irene", "marconi")')
    # parser.add_argument('--dyn', '-d', action='store_true', help='add spectrogram plot to the fitspec module for dynamic fDop analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='print verbose output')

    args = parser.parse_args()

    # required
    subdir    = args.subdir 
    machine   = args.machine
    
    assert machine in ['irene', 'marconi', "altair"], f'machine={machine} not supported. Must be "irene", "marconi" or "altair".'
                
    # optional
    modules   = args.modules
    isims    = args.isims
    isims    = 'all' if isims == 'all' else eval(isims) if isims != [] else []
    verbose   = args.verbose

    from FW2D.gui import FitSpec
    kwargs = dict(verbose=verbose)
    modules_dict = {
        'fit': (FitSpec, (subdir, machine, isims), kwargs, 'Fit Spectra'),
    }


    module_windows = []
    module_titles = []
    for module in modules:
        mod, args, kwargs, title = modules_dict[module]
        module_windows.append(mod(*args, **kwargs))
        module_titles.append(title)
        
    
    window = MainWindow(module_windows, module_titles)
    

    window.show()
    app.exec_()
    
if __name__ == "__main__":
    
    main()
    pass

# %%

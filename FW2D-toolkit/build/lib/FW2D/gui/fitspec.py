#%%
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QShortcut, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QColor,QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtCore

from pyqtplotlib.pltwrapper import AxesWidget
from pyqtplotlib import share_axes

import warnings
import copy
import numpy as np
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)
from FW2D.gui.helper_widgets import WidgetNavigator, ExcludeROIAxesWidget

from FW2D.io.interface import DataInterface
from matlabtools import Struct
from FW2D.processing.sigprocessing import (init_specobjs, OutputWrapper, 
                                          perform_specobj_fits, get_fDop_from_fit_results,
                                          show_estimate_with_errorbars)
class FitSpecWidgetNavigator(WidgetNavigator):
    def __init__(self, widgets, show_first=0, type='list', descriptions=None):
        super().__init__(widgets, show_first, type, descriptions)
    
    def switch_widget(self, index):
        super().switch_widget(index)
        

        # Set the focus to the newly displayed axes widget
        # self.widgets[0,self.current_widget_index].setFocus()
        

class FitSpec(QtWidgets.QWidget):
    def __init__(self, subdir, isims=[], sharex=True, sharey=True, use_mask=False, parent=None, verbose=False):
        super().__init__(parent)
        
        if not verbose:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        self.subdir = subdir
        self.isims = isims
        self.workers = [] # Keep references to the workers
        self.use_mask = use_mask
        self.verbose = verbose
        
        self._setup_spectra(isims) # initializes self.specobjs

        N_plots = len(self.specobjs)
        
        self._setup_ui_elements(N_plots, sharex, sharey)
        self._setup_ui_interaction()  # connects the signals for validating/rejecting fits, etc.
        
        
        specobjs_to_fit = [s for s in self.specobjs if not hasattr(s, 'fit_params')]
        
        self._run_fits(specobjs_to_fit)  # performs fits in Qthread
        # self._update_all_fits(reset_validation=False)
            
        self._load_signal() # load and display the signal I/Q traces on the rhs panel using a Qthread
        
        # shortcut for toggling visibility of data panel
        self.toggle_panel_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        self.toggle_panel_shortcut.activated.connect(self.toggle_panel)
        
        # shorcuts for rejecting (R) or validating (V) fits
        self.reject_fit_shortcut = QShortcut(QKeySequence("R"), self)
        self.reject_fit_shortcut.activated.connect(self._reject_fit)
        self.reject_fit_shortcut = QShortcut(QKeySequence("V"), self)
        self.reject_fit_shortcut.activated.connect(self._validate_fit)
         
        # shortcut for recentering plot using Escape key (pass the key event on to the current ax in focus)
        self.zoom_horizontal_shortcut = QShortcut(QKeySequence("Shift+Esc"), self)
        self.zoom_horizontal_shortcut.activated.connect(self._on_zoom_horizontal)
        
        # shortcut for re-trying the fit using the existing estimate as initial guess:
        self.refit_with_init_guess_shortcut = QShortcut(QKeySequence("Shift+F"), self)
        self.refit_with_init_guess_shortcut.activated.connect(self._on_refit_with_init_guess)
        
        # shortcut for saving state
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self._export)
        
        # shortcut for showing detailed spectrogram option (requires spectrogram module)
        
        # shortcut for applying current spectrum mask to all other ax elements of the current fitspec module:
        self.apply_mask_to_other_axes =  QShortcut(QKeySequence("Ctrl+M"), self)
        self.apply_mask_to_other_axes.activated.connect(self._apply_mask_to_other_axes)
        

        

    def toggle_panel(self, panel_index=1):
        # Toggle the visibility of the data panel
        panel_container = self.navigator.widgets_container[panel_index]
        panel_container.setVisible(not panel_container.isVisible())
        
    def _setup_ui_elements(self, N_plots, sharex=True, sharey=True):
        
        # initialize the AxesWidgets containing the main plots (spectrum + fit)
        # roi_signal = QShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_E, self).activated # unused for the moment
        self.axs = [ExcludeROIAxesWidget(roi_type='rect', roi_signal=None) for i in range(N_plots)]
        # sync axes:
        share_axes(self.axs, axis='x') if sharex else None
        share_axes(self.axs, axis='y') if sharey else None
        
        # initialize a panel (to the right-hand side of the plots) that can be toggled on/off
        self.rhs_panel = []
        
        # the panel will have a sub-panel for data display/interaction:
        self.data_panels = [FitPanel() for i in range(N_plots)]
        
        
        # and a sub-panel for displaying the signal I/Q traces:
        self.axs_signal = [ExcludeROIAxesWidget(roi_type='linear') for i in range(N_plots)]
        self.axs_spec   = [AxesWidget() for i in range(N_plots)]
        share_axes(self.axs_signal, axis='y')
        share_axes(self.axs_spec  , axis='y')
        
        for ax_sig, ax_spec in zip(self.axs_signal, self.axs_spec):
            
            ax_sig.hover_label.hide()
            # ax_spec.hover_label.hide()
            
            share_axes([ax_sig], axis='x')
            share_axes([ax_spec], axis='x')
            
            # to reduce whitespace, we remove the x-axis labels for the signal and Dalpha plots:
            ax_spec.set_xticklabels([])
            
            ax_sig.set_xlabel('Time [s]')
            ax_sig.set_ylabel('')
            ax_spec.set_xlabel('')
            # ax_spec.set_ylabel('') # will be assigned by the spectrogram display method
        
        # here, we combine the data panel and the signal panel into a single container (to be added to the splitter of the navigator, see below)
        for i, (data_panel, ax_sig, ax_spec, ax) in enumerate(zip(self.data_panels, self.axs_signal, self.axs_spec, self.axs)):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            
            ax.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)         
            
            layout.addWidget(data_panel)
            layout.addWidget(ax_spec)
            layout.addWidget(ax_sig)
            container.data_panel = data_panel
            container.ax_sig = ax_sig
            # for easier referencing, let's associate each panel to the respective ax (plot):
            ax.data_panel = data_panel
            ax.ax_sig = ax_sig
            ax.ax_spec = ax_spec
            ax_spec.ax = ax
            data_panel.ax = ax
            ax_sig.ax = ax
            self.rhs_panel.append(container)
            
            
            ax_sig.set_xlabel('Time [s]')
            ax_sig.set_ylabel('')

            
            # ax.hover_label.hide()
        
        ### set up a widget navigator to navigate through the different plots/panels associated with each freq step:
        self.navigator = FitSpecWidgetNavigator(
            [self.axs, self.rhs_panel], type='list', descriptions=[f'{i}' for i in self.ifreqs])
        self.navigator.navigator.currentRowChanged.connect(
            self._update_curr_focus)        
        self._update_curr_focus(0)
        
        # store the initial background color:
        self.neutral_background_color = self.axs[0].backgroundBrush().color()       
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.navigator)
              
    def _update_curr_focus(self, index):
        self.curr_specobj = self.specobjs[index]
        self.curr_ax      = self.axs[index]
        self.curr_data_panel = self.data_panels[index]

        
    def _setup_spectra(self, isims, sharex=True, sharey=True):

        self.data_interface = DataInterface(self.subdir)
        isims = self.data_interface._get_sim_choise(isims)
        self.isims = isims
        self.output_wrapper = OutputWrapper(self.data_interface) # automatically loads the existing data from file if any
           
        # initialize the specobjs:
        specobjs = init_specobjs(self.data_interface, isims=self.isims,)
        
        # check if some of the specobjs already have been treated (validated or rejected), in which case they are overwritten:
        isims_treated = self.output_wrapper.header.isims_treated
        if len(isims_treated) > 0:
            if self.verbose:
                print(f'Found {len(self.output_wrapper.header.isims_treated)} specobjs already treated.')
                
            for i in isims_treated:
                if i in isims:
                    k = (np.where(i==isims))[0][0]
                    specobjs[k] = self.output_wrapper.merged_specobj.specobjs[i-1]
        
        self.output_wrapper.update_specobjs(specobjs)
        
        self.specobjs = specobjs


    def _run_fits(self, specobjs=None):
        
        if specobjs is None:
            specobjs = self.specobjs
            
        worker = Worker(specobjs, include_masks = [None]*len(specobjs) )
        worker.fitPerformed.connect(lambda: self._update_all_fits(reset_validation=False))
        worker.finished.connect(self._cleanup_worker)
        self.workers.append(worker)
        worker.start()
            
        
    def _setup_ui_interaction(self):
        
        # if not hasattr(self, 'specobjs'):
        #     self._setup_spectra()
        
        from FW2D.processing.sigprocessing import make_title
        
        for i, (s, ax, ax_sig, data_panel) in enumerate(zip(self.specobjs, np.array(self.axs).flatten(), self.axs_signal, self.data_panels )):

            
            # Pass the current ax and s as default arguments to the lambda
            ax.update_fit_key_pressed.connect(lambda ax=ax, specobj=s, data_panel=data_panel: self._perform_fit(
                ax=ax, specobj=specobj, data_panel=data_panel, include_mask=ax.mask))

            ax.roi.sigRegionChangeFinished.connect(
                lambda ax=ax, specobj=s: self._reset_fit_validation(ax, specobj))
            
                
            # connect the data panels signals:
            for curve_type in ['odd', 'gaussian', 'lorentzian', 'taylor']:
                checkbox = getattr(data_panel, f'checkbox_{curve_type}')
                checkbox.check_box.clicked.connect(
                    lambda state, ax=ax, specobj=s, data_panel=data_panel: self._on_checkboxes_clicked(ax, specobj, data_panel))
                
            textboxes = {
                'fDop': data_panel.textbox_fDop,
                # 'dfDop': data_panel.textbox_dfDop,
                'fDop_max': data_panel.textbox_fDop_max,
                'fDop_min': data_panel.textbox_fDop_min,
            }
            for key,textbox in textboxes.items():
                textbox.text_box.returnPressed.connect(
                    lambda specobj=s, data_panel=data_panel, textbox=textbox.text_box, key=key: self._on_textbox_returnPressed(specobj, data_panel, textbox, key))

            make_title(ax, self.data_interface, s.header.ifreq)
            
            
            # signal plot interaction:
            ax_sig.roi.sigRegionChangeFinished.connect(
                lambda roi=ax_sig.roi, ax_sig=ax_sig, specobj=s: self._update_signal_plot(ax_sig, specobj, roi))
            
    def _perform_fit(self, ax, specobj, data_panel, include_mask=None, **kwargs):
        
        # shortcuts
        s = specobj  
        
        if include_mask is not None:
            include_mask = ~include_mask# Todo: definitions clash: should better call one of them exclude_mask instead
            
        p0 = kwargs.pop('p0', None)
        
        # check if signal mask has changed, in which case we need to reinitialize the specobj altogether (before fitting):
        curr_mask = ~ax.ax_sig.mask
        if not np.array_equal(curr_mask, s.signal_include_mask):
            s.signal_include_mask = np.copy(curr_mask)
            reinitialize = True
        else:
            reinitialize = False

            
        perform_specobj_fits(specobj, include_mask, p0, reinitialize, s.signal_include_mask, self.verbose)
        self._update_fit(ax, specobj, data_panel, reset_panel=True, reset_validation=True, **kwargs)

            
    def _update_fit(self, ax, specobj, data_panel, reset_panel=False, reset_validation=True, **kwargs):
        
        # update the include_mask:
        if specobj.include_mask is not None:
            ax.mask = ~specobj.include_mask
        
        if not 'legend' in kwargs:
            kwargs['legend'] = True
        if not 'lw' in kwargs:
            kwargs['lw'] = 2
        
        if hasattr(ax, 'plot_dict'):
            for item in ax.plot_dict.values():
                ax.removeItem(item)
        
        from FW2D.processing.sigprocessing import show_spec
        ax.plot_dict = show_spec(specobj, ax=ax, **kwargs)
        
        
        # self.lines_to_hide_when_interacting = ['loddPSD', 'lfit_odd', 'lfit_gaussian', 'lfit_lorentzian', 'lfit_taylor']
        
        if 'loddPSD' in ax.plot_dict:
            loddPSD = ax.plot_dict['loddPSD']
            loddPSD.hide()
        

        ax.data = ax.plot_dict['lrawPSD'].getData()
        ax.roi_data_item = ax.plot_dict['lfittedPSD']
               
        self._update_panel(specobj, data_panel, reset=reset_panel)
        
        if reset_validation:
            self._reset_fit_validation(ax, specobj)
        
            
    def _update_all_fits(self, **kwargs):
        for i, (s, ax, data_panel) in enumerate(zip(self.specobjs, np.array(self.axs).flatten(), self.data_panels )):
            self._update_fit(ax, s, data_panel, **kwargs)
            self._update_fit_validation(ax, s)
            
    def _update_panel(self, specobj, data_panel, reset=True):
        
        if hasattr(specobj, 'fit_params'):
            fit_params = specobj.fit_params
            fDop_results = get_fDop_from_fit_results(fit_params, specobj.xscale)

            
            # Todo: make a function _get_selected_checkboxes() that returns the selected checkboxes or fDop_results_selected() that shall be called in _on_textbox_returnPressed() to update the text boxes
            if hasattr(specobj, 'odd_even_threshold'):
                f_cog = specobj.f_cog if (specobj.power_odd / specobj.power_even > specobj.odd_even_threshold) else None
            else:
                f_cog = None
                

            if hasattr(specobj, 'fDop_checkboxes_state'):
                checked_state = specobj.fDop_checkboxes_state
            else:
                checked_state = {
                    'cog': False,
                    'odd': True,
                    'gaussian': True,
                    'lorentzian': True,
                    'taylor': True,
                }
                specobj.fDop_checkboxes_state = checked_state
            
            # update the labels:
            for curve_type in ['cog', 'odd', 'gaussian', 'lorentzian', 'taylor']:
                checkbox = getattr(data_panel, f'checkbox_{curve_type}')
                
                # if reset:
                    
                _fDop = fDop_results.get(f'fDop_{curve_type}', None) if curve_type != 'cog' else f_cog

                if _fDop is None or np.isnan(_fDop):
                    checkbox.update(description=f'{curve_type} -')
                    checkbox.setChecked(False)
                else:
                    if curve_type=='cog': # ignore the cog for now
                        checkbox.setChecked(False)
                    else:
                        checkbox.update(description=f'{curve_type} {_fDop / 1e3:.1f}') # display in KHz
                        checkbox.setChecked(checked_state[curve_type]) # by, default, check the box if the fit converged (meaning it will be used for the median), unless the user unchecked it before
                    
            
            
            # get the estimate according to the checkbox checked states:
            fDop_results_selected = self._get_fDop_results_selected(specobj, data_panel, fit_params)
            
            if reset or not hasattr(specobj, 'fDop_max'):
                specobj.fDop_max = fDop_results_selected['fDop_max']
                specobj.fDop_min = fDop_results_selected['fDop_min']
            if reset or not hasattr(specobj, 'fDop'):
                # update the final fDop and dfDop values:             
                specobj.fDop = fDop_results_selected['fDop']
                specobj.dfDop = specobj.fDop_max - specobj.fDop_min
            
            self._update_fDop_text_boxes(data_panel, specobj, fDop_results_selected)
            self._update_odd_even_label(data_panel, specobj.power_odd / specobj.power_even)
            
            self._update_estimate_with_errorbar(specobj, data_panel.ax)
        else:
            warnings.warn('specobj has no attribute "fit_params"')
    
       
    
    def _update_signal_plot(self, ax_sig, specobj, roi=None):

        # shorthands:
        ax, s = ax_sig, specobj
        
        # not needed anymore, since we now have the signal mask implemented:
        # if hasattr(s.header, 'twindow'):
        #     ax.axvline(s.header.twindow[0], color='k', linestyle='--')
        #     ax.axvline(s.header.twindow[1], color='k', linestyle='--')
        
        if not hasattr(s, 'signal_include_mask'):
            s.signal_include_mask = np.ones_like(ax.t, dtype=bool)
            # mask out the edges of the signal (set dtstart and dtend to 0 to keep the whole signal):
        
        if not hasattr(ax, 'plot_dict'): # plot initialization
        
        
            I = I 
            Q = Q 
            Z = I * np.exp(1j * Q)
            
                        
            lI = ax.plot(t, I, color='r', alpha=0.6, label='I')
            lQ = ax.plot(t, Q, color='b', alpha=0.6, label='Q')
            
            labsZ_masked = ax.plot(t, np.abs(Z), color='k', alpha=1, label='|I + i Q|')
            ax.plot_dict = {'lI': lI, 'lQ': lQ, 'labsZ_masked': labsZ_masked}
        
            ax.data = (t,np.abs(Z)) #  ax.plot_dict['labsZ'].getData()
            ax.roi_data_item = ax.plot_dict['labsZ_masked']
        
            ax.mask = ~np.copy(s.signal_include_mask)
            ax.update_mask(ax.mask)
            
            # indicate the rms:
            ax.axhline(np.mean(I) - np.std(I),  color='r', linestyle='--')
            ax.axhline(np.mean(Q) - np.std(Q),  color='b', linestyle='--')
            ax.axhline(np.mean(I) + np.std(I),  color='r', linestyle='--')
            ax.axhline(np.mean(Q) + np.std(Q),  color='b', linestyle='--')
        
            ax.legend()
        
        else: # only an update is needed (mask change)
            # perform inverse downsample:
            
            pass
            
            
    def _update_all_signal_plots(self, axs_signal=None, specobjs=None):
        if axs_signal is None:
            axs_signal = self.axs_signal
        if specobjs is None:
            specobjs = self.specobjs
        for ax_sig,s in zip(axs_signal, specobjs):
            self._update_signal_plot(ax_sig, s)
            
                
    def _apply_mask_to_other_axes(self):
        curr_ax = self.curr_ax
        curr_s = self.curr_specobj
        mask = np.copy(curr_ax.mask)
        
        
        for ax, s in zip(self.axs, self.specobjs): 
            if ax==curr_ax:
                continue
            
            ax.update_mask(mask)
            
            # self._update_fit(ax, s, ax.data_panel)
            # self._update_fit_validation(ax, s)
        

    def _update_fDop_text_boxes(self, data_panel, specobj, fDop_results_selected):
        

            e = Struct(fDop_results_selected) # automatic stimates
            a = specobj # actual values (stored to file, can be modified manually)
            
            data_panel.textbox_fDop.update(description=f'Estimate (median: {e.fDop/1e3:.1f})', value=f'{a.fDop/1e3:.1f}')
            data_panel.textbox_dfDop.update(description=f'Error (max - min: {e.dfDop/1e3:.1f})', value=f'{a.dfDop/1e3:.1f}')
            data_panel.textbox_fDop_max.update(description=f'max: {e.fDop_max/1e3:.1f}', value=f'{a.fDop_max/1e3:.1f}')
            data_panel.textbox_fDop_min.update(description=f'min: {e.fDop_min/1e3:.1f}', value=f'{a.fDop_min/1e3:.1f}')
            
            
    def _update_odd_even_label(self, data_panel, odd_even_ratio):
        data_panel.odd_even_ratio_label.setText(f'Odd/even ratio:{odd_even_ratio:.3f}')
        # data_panel.odd_even_ratio_label.update(description=f'Odd/even ratio:{odd_even_ratio:.3f}')
        
    # def _on_odd_even_threshold_changed(self, ax, specobj, data_panel, **kwargs):
    #     new_threshold = self.data_panel.odd_even_ratio_label.value()
    #     for specobj in self.specobjs:
    #         specobj.odd_even_threshold
    #     self._update_panel(specobj, data_panel, reset=False, **kwargs)
    #     self._reset_fit_validation(ax, specobj)
            
    def _on_checkboxes_clicked(self, ax, specobj, data_panel, **kwargs):

        # update the checked state:
        for curve_type in ['cog', 'odd', 'gaussian', 'lorentzian', 'taylor']:
            checkbox = getattr(data_panel, f'checkbox_{curve_type}')
            specobj.fDop_checkboxes_state[curve_type] = checkbox.isChecked()
            
        # update the labels:
        self._update_panel(specobj, data_panel, reset=True, **kwargs)
        self._reset_fit_validation(ax, specobj)
        
    def _get_fDop_results_selected(self, specobj, data_panel, fit_params):
        
        # fit_params = specobj.fit_params
        fit_params_selected = copy.deepcopy(fit_params)
        # fDop_results = get_fDop_from_fit_results(fit_params, specobj.xscale)
            
        for curve_type in ['cog', 'odd', 'gaussian', 'lorentzian', 'taylor']:
            checkbox = getattr(data_panel, f'checkbox_{curve_type}')               
            
            # remove fit results that are not checked or defined:
            if not checkbox.isChecked():
                fit_params_selected.pop(curve_type, None)
        
        fDop_results_selected = get_fDop_from_fit_results(
            fit_params_selected, specobj.xscale)
        
        return fDop_results_selected

    def _update_fit_validation(self, ax, specobj):
        self._validate_fit(ax, specobj) if specobj.validated==1 else self._reject_fit(ax, specobj) if specobj.validated==-1 else self._reset_fit_validation(ax, specobj)

    def _reset_fit_validation(self, ax, specobj):
        specobj.validated = 0
        # change the plot's frame to transparent white:
        if hasattr(ax, 'setBackground'):
            ax.setBackground(self.neutral_background_color)
        self.output_wrapper.update_specobjs([specobj])
        
    def _hide_lines_during_roi_change(self, ax, specobj):
        for line_name in self.lines_to_hide_when_interacting:
            if hasattr(ax, 'plot_dict'):
                if line_name in ax.plot_dict:
                    # ax.plot_dict[line_name].hide()
                    ax.removeItem(ax.plot_dict[line_name])
            else:
                print('ax has no attribute "plot_dict"')
                
    def _show_lines_after_roi_change(self, ax, specobj):
        for line_name in self.lines_to_hide_when_interacting:
            if hasattr(ax, 'plot_dict'):
                if line_name in ax.plot_dict:
                    ax.plot_dict[line_name].show()
            else:
                print('ax has no attribute "plot_dict"')
                
    def _validate_fit(self, ax=None, specobj=None, **kwargs):
        
        if ax is None:
            ax = self.curr_ax
        if specobj is None:
            specobj = self.curr_specobj
            
        specobj.validated = 1
        # change the plot's frame to transparent green:
        ax.setBackground(QColor(0,255,0,20))
        self.output_wrapper.update_specobjs([specobj])
    
    def _reject_fit(self, ax=None, specobj=None, **kwargs):
        
        if ax is None:
            ax = self.curr_ax
        if specobj is None:
            specobj = self.curr_specobj
            
        specobj.validated = -1
        specobj.fDop = np.nan
        specobj.dfDop = np.nan
        # self._update_fDop_text_boxes(ax.data_panel, specobj)
        # change the plot's frame to transparent red:
        ax.setBackground(QColor(255, 0, 0, 20))
        self.output_wrapper.update_specobjs([specobj])
        
    def _on_refit_with_init_guess(self):
        
        ax = self.curr_ax
        specobj = self.curr_specobj
        data_panel = ax.data_panel
        
        # init guess based on the current estimate
        
        # To go further: use the Taylor fit (which is usually quite good) to initialize the Gaussian and Lorentzian fits for a second fit if necessary
        
        p0 = [1.0, specobj.fDop / specobj.xscale, 5.0]
        print(f'retrying fit with init guess: {p0}')
        self._perform_fit(
                ax=ax, specobj=specobj, data_panel=data_panel, include_mask=ax.mask, p0=p0)
        
        # self._update_fDop_text_boxes(data_panel, specobj)
    
    
    def _on_textbox_returnPressed(self, specobj,data_panel, textbox, key):
        
        old_val = getattr(specobj, key)
        try:
            new_val = float(eval(textbox.text())) * 1e3
        except Exception as e:
            if self.verbose:
                print(f'skipping update of {key} due to error:\n{e}')
            return
        setattr(specobj, key, new_val)
        
        specobj.dfDop = specobj.fDop_max - specobj.fDop_min
        
        fDop_results_selected = self._get_fDop_results_selected(specobj, data_panel, specobj.fit_params)
        self._update_fDop_text_boxes(data_panel, specobj, fDop_results_selected)
        # self._update_panel(specobj, data_panel, reset=False)
        
        if self.verbose:
            print(f'updating {key} from {old_val/1e3:.1f} to {new_val/1e3:.1f} kHz') 
            
        self._update_estimate_with_errorbar(specobj, data_panel.ax)
        
    
    def _update_estimate_with_errorbar(self, specobj, ax):
        
        _plot_dict = show_estimate_with_errorbars(ax, specobj, lw=2)
        
        for key in ['fDop_vline', 'dfDop_hline']:
            item = ax.plot_dict.get(key, None)
            if item is not None:
                ax.removeItem(item)
        
        ax.plot_dict.update(_plot_dict)
          
        
    def _export(self):
        
        self.output_wrapper.update_specobjs(self.specobjs)
        self.output_wrapper.export()
        print(f'Saved to fit results to {self.output_wrapper.outpath}')
        
        
    def _load_signal(self):
        
        worker = WorkerSignal(self)
        worker.data_loaded.connect(self._update_all_signal_plots)
        worker.finished.connect(self._cleanup_worker)
        self.workers.append(worker)
        worker.start()
         
    def _toy_data(self):
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 4, 9, 16]
        self.axs[0,0].plot(x,y, color='b', linestyle='--', marker='x', label='data')
        
        
    def OnAboutToQuit(self):
        if self.verbose:
           print('Quitting QApplication cleanly...')
        # self.output_wrapper.export()
        
    def _test_functions(self):
        ax, s = self.axs[0,0], self.specobjs[0]
        self._validate_fit(ax, s)
        # for i, (s, ax) in enumerate(zip(self.specobjs, self.axs.flatten())):
     
    def _cleanup_worker(self):
        sender = self.sender()
        self.workers.remove(sender)
        sender.deleteLater()
        
    def _on_zoom_horizontal(self, zoom_factor=5):
        ax = self.curr_ax
        # get current limits:
        xmin, xmax = ax.get_xlim()
        xcenter = np.mean([xmin, xmax])
        xmin_new = xcenter - (xcenter - xmin) / zoom_factor
        xmax_new = xcenter - (xcenter - xmax) / zoom_factor
        ax.set_xlim(xmin_new, xmax_new)

class InfoPanel(QtWidgets.QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        from PyQt5.QtWidgets import (QGroupBox, QSpacerItem, QSizePolicy)
        from PyQt5.QtGui import QIntValidator, QDoubleValidator
        from FW2D.gui.helper_widgets import CustomCheckBox, CustomTextBox

        # Create the main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create a group box for related UI items
        self.group_box = QGroupBox("Doppler shift [kHz]")
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)

        
        # Add a spacer item to push all widgets to the top
        spacer_item = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        group_layout.addItem(spacer_item)
        
        # Set the group layout to the group box
        self.group_box.setLayout(group_layout)
        
        
        # create a second group box:
        self.group_box_2 = QGroupBox("Info")
        self.group_box_2.setLayout(group_layout)
        
        

        # Add the group box to the main layout
        main_layout.addWidget(self.group_box)

        # If you have other groups, create more QGroupBoxes here and add them to main_layout
        # ...

        # Set the main layout to the DataPanel
        self.setLayout(main_layout)
        
class FitPanel(QtWidgets.QWidget):
    
    TEXTBOX_MIN_SIZE = (50,10)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        from PyQt5.QtWidgets import (QGroupBox, QSpacerItem, QSizePolicy)
        from PyQt5.QtGui import QIntValidator, QDoubleValidator
        from FW2D.gui.helper_widgets import CustomCheckBox, CustomTextBox

        # Create the main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        
        # Create a group box for related UI items
        self.group_box = QGroupBox("Doppler shift [kHz]")
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)
        self.group_box.setLayout(group_layout)
        
        # create a second group box in the exact same way:
        self.group_box_2 = QGroupBox("")
        group_layout_2 = QVBoxLayout()
        group_layout_2.setContentsMargins(0, 0, 0, 0)
        group_layout_2.setSpacing(0)
        self.group_box_2.setLayout(group_layout_2)

        # Add check boxes for each fit:
        for curve_type in ['cog', 'odd', 'gaussian', 'lorentzian', 'taylor']:
            checkbox = CustomCheckBox(description=curve_type)
            setattr(self, f'checkbox_{curve_type}', checkbox)
            group_layout.addWidget(checkbox)

        
        # Add a spacer item to push all widgets to the top
        spacer_item = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        group_layout.addItem(spacer_item)
        
        # add a label:
        validator = QDoubleValidator(0, 1, 3)

        self.odd_even_ratio_label = QLabel('Odd/even ratio:')
        group_layout_2.addWidget(self.odd_even_ratio_label)
        # add one checkbox for each fit:
        
        # Add a text box for the final Doppler shift estimate, and the associated error
        fDop_validator = QDoubleValidator(-1e4, 1e4, 2)
        kwargs = dict(default_value=0, validator=fDop_validator,
            min_size=FitPanel.TEXTBOX_MIN_SIZE)
        self.textbox_fDop = CustomTextBox(
            description='', **kwargs)
        self.textbox_dfDop = CustomTextBox(
            description='Error (max - min: ):', **kwargs)
        group_layout_2.addWidget(self.textbox_fDop)
        group_layout_2.addWidget(self.textbox_dfDop)
        
        # add a horzontal layout with two textboxes for the min/max estimates:
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        kwargs = dict(default_value=0,
            validator=fDop_validator, min_size=FitPanel.TEXTBOX_MIN_SIZE)
        self.textbox_fDop_min = CustomTextBox(description='min:', **kwargs)
        self.textbox_fDop_max = CustomTextBox(description='max:', **kwargs)
        hbox.addWidget(self.textbox_fDop_min)
        hbox.addWidget(self.textbox_fDop_max)
        group_layout_2.addLayout(hbox)
        
        
        # Add also a spacer item to push all widgets to the top
        spacer_item_2 = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        group_layout_2.addItem(spacer_item_2)
        

        # Add the group box to the main layout
        main_layout.addWidget(self.group_box)
        main_layout.addWidget(self.group_box_2)

        # Set the main layout to the DataPanel
        self.setLayout(main_layout)
        # self.setMinimumSize(200, 230)
        # self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        
    def add_checkbox_for_plotitem(self, plotitem):
        from FW2D.gui.helper_widgets import CustomCheckBox
        checkbox = CustomCheckBox(description=plotitem.name())
        setattr(self, f'checkbox_{plotitem.name()}', checkbox)
        self.group_box_2.layout().addWidget(checkbox)



class Worker(QThread):
    # Signal for cleanup
    finished = pyqtSignal()
    # Signal to emit when fits are done
    fitPerformed = pyqtSignal()

    def __init__(self, specobjs, include_masks, parent=None):
        super().__init__(parent)  # Call the __init__ of the QThread class
        self.specobjs = specobjs
        self.include_masks = include_masks
        self.parent = parent
        
    def run(self):
        
        for s,mask in zip(self.specobjs, self.include_masks):
            perform_specobj_fits(s, mask)
        
        self.fitPerformed.emit()
        
        self.finished.emit()
        
class WorkerSignal(QThread):

    # Signal for cleanup
    finished = pyqtSignal()
    # Signal to emit loaded data
    data_loaded = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)  # Call the __init__ of the QThread class
        self.parent = parent
        
        
    def run(self):
        
        from FW2D.processing.sigprocessing import get_normalized_complex_signal
        
        
        for (ax,isim) in zip(self.parent.axs_signal, self.parent.isims):
            t, I,Q = self.parent.data_interface.get_signal(isim)
            ax.t = t; ax.I = I; ax.Q = Q;
            
        self.data_loaded.emit()
        self.finished.emit()
        
        
        
#%%

if __name__ == "__main__":
    import sys
    
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    
    fitspec = FitSpec('mixed_advection_2', isims = [1])
    # fitspec = FitSpec(85556,'tcv',9,1, ifreqs=[1,10,11], dyn=True, use_mask=True, verbose=False)
    # fitspec = FitSpec(81178,'tcv',14,2, ifreqs=[7,8], verbose=False)
    # fitspec = FitSpec(78541,'tcv',5,1, ifreqs=[5,6,7], verbose=False)
    # fitspec = FitSpec(79357,'tcv',8,1, ifreqs=[8])
    # fitspec = FitSpec(78541,'tcv',3,1, ifreqs=[10])
    window.setCentralWidget(fitspec)

    app.aboutToQuit.connect(fitspec.OnAboutToQuit)
    
    import signal
    def signal_handler(signum, frame):
        # Perform clean-up tasks here
        print("Termination signal received, cleaning up.")
        app.aboutToQuit.emit()
        signal.SIG_DFL
        sys.exit(0)

    # Register the signal handler for keyboard interrupt (Ctrl+C + Enter)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # fitspec._test_functions()
    window.show()
    app.exec_()

    # %%

# %%

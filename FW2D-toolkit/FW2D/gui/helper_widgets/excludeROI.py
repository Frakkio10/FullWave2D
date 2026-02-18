from PyQt5 import QtCore
from pyqtplotlib.interaction import ROIAxesWidget
from pyqtplotlib.pltwrapper import AxesWidget
from PyQt5.QtCore import pyqtSignal
import numpy as np

class ExcludeROIAxesWidget(AxesWidget, ROIAxesWidget):

    update_fit_key_pressed = pyqtSignal()
    validate_fit_key_pressed = pyqtSignal()
    reject_fit_key_pressed = pyqtSignal()
    save_key_pressed = pyqtSignal()

    def __init__(self, roi_signal=None, parent=None, roi_type='rect'):
        """Plot widget with an ROI that can be used to exclude data points from the plot.
        """

        super().__init__(parent=parent, roi_type=roi_type)

        # Accept focus by both tabbing and clicking
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.setMouseTracking(True)

        self.mask = None

        self.set_title('')

        self.data = None
        
        self.shiftPressed = False

        # self.update_fit_key_pressed.connect(self.onUpdateFitKeyPressed)


        
        
    def onROIChanged(self):
        
        inverted = self.shiftPressed
        x, y = self.data
        if self.roi_type == 'rect':
            bounds = self.roi.parentBounds()
            x1, y1, x2, y2 = bounds.left(), bounds.top(), bounds.right(), bounds.bottom()
            mask = (x > x1) & (x < x2) & (y > y1) & (y < y2)
        elif self.roi_type == 'linear':
            x1, x2 = self.roi.getRegion()
            mask = (x > x1) & (x < x2)
        self.update_mask(mask, inverted=inverted)

    def update_mask(self, mask, inverted=False):

        x, y = self.data

        if self.mask is None:
            self.mask = np.zeros_like(x, dtype=bool)
        else:

            if not inverted:  # append to existing mask
                self.mask[mask] = True
                # self.mask = np.hstack([self.mask, mask])
            else:  # remove from existing mask
                self.mask[mask] = False

        # self.set_data(x[~mask], y[~mask])
        self.roi_data_item.setData(x[~self.mask], y[~self.mask])

    def plot(self, *args, **kwargs):
        """Override the default behavior of plot(), which would create a new item.
        Instead, the data is set to the existing item, if the item is provided as a keyword argument."""
        item = kwargs.pop('item', None)
        new_item = super().plot(*args, **kwargs)
        if item is not None:
            item.setData(*new_item.getData())
            # remove new item:
            self.removeItem(new_item)
            return item
        else:
            return new_item

    def keyPressEvent(self, event):
        
        if event.key() == QtCore.Qt.Key_Shift:
            self.shiftPressed = True
            
        
        # Check if the event is an auto-repeat event
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        # print(f"Event received by widget: {self}\n")
        # if hasattr(self, 'id'):
        #     print(self.id)

        # override the default behavior of the base class (if any)

        if self.hasFocus() and event.key() == QtCore.Qt.Key_F:
            self.update_fit_key_pressed.emit()

        elif self.hasFocus() and (event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter):
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self.reject_fit_key_pressed.emit()
            else:
                self.validate_fit_key_pressed.emit()

        elif event.key() == QtCore.Qt.Key_S and event.modifiers() == QtCore.Qt.ControlModifier:
            self.save_key_pressed.emit()

        else:
            # in all other cases, call the default behavior of the base class
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Shift:
            self.shiftPressed = False    
        super().keyReleaseEvent(event)
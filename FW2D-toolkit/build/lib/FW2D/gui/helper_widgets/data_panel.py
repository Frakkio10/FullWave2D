#%%
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSizePolicy,
                             QLineEdit, QCheckBox, QApplication, QMainWindow)
import sys

from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
from PyQt5.QtWidgets import QHBoxLayout

class CustomTextBox(QWidget):
    
    def __init__(self, description='', default_value='', validator=None, min_size=None):
        
        super().__init__()
    
        self.text_box = QLineEdit()
        self.text_box.setValidator(validator)  # Allows only floats with 2 decimal places
        self.text_box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        if min_size is not None:
            self.text_box.setMinimumSize(*min_size)
        
        # Create a horizontal layout for the label and the text box
        hbox = QHBoxLayout()

        # Add a label with a description
        self.label_description = QLabel(description)
        self.label_description.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        hbox.addWidget(self.label_description)

        # Add the text box to the horizontal layout
        hbox.addWidget(self.text_box)
        
        self.update(default_value)
        
        self.setLayout(hbox)
        
    def update(self, value=None, description=None):
        if value is not None:
            self.text_box.setText(str(value))  # Update the text box with a new value
        if description is not None:
            self.label_description.setText(description)
            


class CustomCheckBox(QWidget):

    def __init__(self, description='', is_checked=False):
        super().__init__()

        self.check_box = QCheckBox()

        # Set the initial state of the checkbox
        self.check_box.setChecked(is_checked)
        self.check_box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        # Create a horizontal layout for the label and the checkbox
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)

        # Add a label with a description
        self.label_description = QLabel(description)
        hbox.addWidget(self.label_description)

        # Add the checkbox to the horizontal layout
        hbox.addWidget(self.check_box)

        self.setLayout(hbox)

    def update(self, is_checked=None, description=None):
        if is_checked is not None:
            self.check_box.setChecked(is_checked)
        if description is not None:
            self.label_description.setText(description)
            
    def isChecked(self):
        return self.check_box.isChecked()
    def setChecked(self, value):
        self.check_box.setChecked(value)


        
class ExampleDataPanel(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the layout
        layout = QVBoxLayout()

        # Set the layout to the QWidget
        self.setLayout(layout)
        
        # Add check boxes for each fit:
        for curve_type in ['gauss', 'lorentz', 'taylor']:
            checkbox = CustomCheckBox(description=curve_type)
            layout.addWidget(checkbox)
        
        
        # Add a text box
        validator = QDoubleValidator(-1e4, 1e4, 2)
        textbox = CustomTextBox(description='Doppler freq.  [kHz]', default_value=0, validator=validator)
        layout.addWidget(textbox)

if __name__ == '__main__':
    
    
    from pyqtplotlib.pltwrapper import AxesWidget
    from FW2D.gui.helper_widgets import WidgetNavigator
        
    app = 0
    app = QApplication(sys.argv)
    
    window = QMainWindow()

    # Initialize plot widgets
    axs = [AxesWidget() for _ in range(3)]
    axs[0].plot([1, 2, 3], [1, 2, 3])
    axs[1].plot([1, 2, 3], [3, 2, 1])
    axs[2].plot([1, 2, 3], [4, 0, 4])
        
    navigator = WidgetNavigator(axs, type='combo')
    
    # Create an instance of the DataPanel
    data_panel = ExampleDataPanel()

    # Add the DataPanel instance to the splitter
    navigator.splitter.addWidget(data_panel)
    
    window.setCentralWidget(navigator)
    window.show()
    sys.exit(app.exec_())

# %%

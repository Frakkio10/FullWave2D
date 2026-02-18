# %%
import sys
from PyQt5.QtWidgets import QSplitter, QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QListWidget
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize 
import numpy as np

class WidgetNavigator(QWidget):
    def __init__(self, widgets, show_first=0, type='list', descriptions=None):
        super().__init__()
        """Create a widget navigator to switch between multiple widgets.
        Args:
        -----
        widgets : list(s) of QWidget, can be a list of lists of QWidget of the same length (the navigator will update the widget within each list simultaneously)
        show_first : int, optional. The index of the widget to show first.
        descriptions : list of str, optional. The descriptions to show in the navigator. If not provided, the widgets will be numbered.
        """
        
        # check input dimensionsality:
        widgets = np.array(widgets, dtype=object)
        
        if widgets.ndim == 1:
            if not len(widgets) > 0:
                raise ValueError('Must provide at least one widget to navigate')
            else:
                widgets = widgets.reshape(1, -1)
        elif widgets.ndim == 2:
            pass
        
        
        if not all(isinstance(widget, QWidget) for widget in widgets.flat):
            raise ValueError('All widgets must be of type QWidget')
        
        if descriptions is None:
            descriptions = [f'{i}' for i in range(widgets.shape[1])]
        else:
            if not len(descriptions) == widgets.shape[1]:
                raise ValueError('Must provide a description for each item in the widget list.')

        # Create the splitter
        self.splitter = QSplitter()
        self.setLayout(QVBoxLayout())  # Set the layout for WidgetNavigator
        self.layout().addWidget(self.splitter)  # Add the splitter to the layout
        
        self.widgets_container = [] # to keep reference to the containers (one for each list of widgets)

        self._add_navigator(type, descriptions)
        self._add_widgets(widgets, show_first)

    def _add_navigator(self, type, descriptions):
        # Create the navigator widget
        if type == 'list':
            
            class CustomListWidget(QListWidget):
                def sizeHint(self):
                    # Specify the desired size hint
                    return QSize(100, self.height())  # Replace 100 with your desired width
            self.navigator = CustomListWidget()
            self.navigator.addItems(descriptions)
            self.navigator.currentRowChanged.connect(self.switch_widget)
            
        elif type == 'combo':
            
            self.navigator = QComboBox()
            self.navigator.addItems(descriptions)
            self.navigator.currentIndexChanged.connect(self.switch_widget)
        else:
            raise ValueError('Navigator type must be "list"')

        # Set the size policy for the navigator to limit its growth
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.navigator.setSizePolicy(sizePolicy)

        # Add the navigator to the splitter
        self.splitter.addWidget(self.navigator)

    def _add_widgets(self, widgets, show_first=0):
        self.current_widget_index = show_first
        self.widgets = widgets
        
        for ilist in range(widgets.shape[0]):

            # Create a container for the plot widgets
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            for widget in widgets[ilist]:
                layout.addWidget(widget)
                widget.hide()

            self.switch_widget(show_first)

            # Add the plot container to the splitter
            self.splitter.addWidget(container)
            
            # keep reference to the container:
            self.widgets_container.append(container)


    def switch_widget(self, index):
        
        # Hide the current widgets from all lists
        for ilist in range(self.widgets.shape[0]):
            self.widgets[ilist][self.current_widget_index].hide()
            
            
        # Show the new widgets from all lists
        self.current_widget_index = index
        for ilist in range(self.widgets.shape[0]):
            self.widgets[ilist][self.current_widget_index].show()
            # Bring the widget to the front
            self.widgets[ilist][self.current_widget_index].raise_()
            
if __name__ == '__main__':
    
    from pyqtplotlib.pltwrapper import AxesWidget
        
    app = 0
    app = QApplication(sys.argv)
    
    window = QMainWindow()

    # Initialize plot widgets
    axs = [AxesWidget() for _ in range(3)]
    axs[0].plot([1, 2, 3], [1, 2, 3])
    axs[1].plot([1, 2, 3], [3, 2, 1])
    axs[2].plot([1, 2, 3], [4, 0, 4])
    second_axs = [AxesWidget() for _ in range(3)]
    second_axs[0].plot([1, 2, 3], [1, 2, 3])
    second_axs[1].plot([1, 2, 3], [3, 2, 1])
    second_axs[2].plot([1, 2, 3], [4, 0, 4])
    third_axs = [AxesWidget() for _ in range(3)]
    third_axs[0].plot([1, 2, 3], [1, 2, 3])
    third_axs[1].plot([1, 2, 3], [3, 2, 1])
    third_axs[2].plot([1, 2, 3], [4, 0, 4])
    
        
    navigator = WidgetNavigator([axs, second_axs], type='list')
    window.setCentralWidget(navigator)
    window.show()
    sys.exit(app.exec_())

# %%

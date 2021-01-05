import matplotlib.pyplot as plt
from nxsdk.utils.plotutils import plotRaster

import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_()) 

class Plotter():
    def __init__(self, figsize=(18,3)):
        self.figsize = figsize

    def plot(self):
        pass

        # TODO
        # create a grid layout of subplots or something;
        # put all registered plots into that grid

    def add_input_plot(self, input):
        """Create a plot of the spike pattern of the input over time."""
        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(input.name)
        plotRaster(input.spike_times)
        plt.xlabel('Time steps')
        plt.ylabel('Input index')
        plt.xlim(0, input.number_of_time_steps)
        plt.title('Source Spikes')
        plt.tight_layout()
        plt.show()
        #a = ScrollableWindow(fig)

    def add_node_plot(self, node):
        """Creates plots for all probes of the node."""
        if (node.probe_current == None or node.probe_voltage == None or node.probe_spikes == None):
            print("Error: Node does not have all necessary probes to plot.")
        else:
            fig = plt.figure(figsize=(18,10))
            fig.suptitle(node.name)

            ax0 = plt.subplot(3,1,1)
            node.probe_current.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Current')
            plt.title('Current')

            ax1 = plt.subplot(3,1,2)
            node.probe_voltage.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Voltage')
            plt.title('Voltage')

            ax2 = plt.subplot(3,1,3)
            node.probe_spikes.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Neuron index')
            plt.title('Spikes')
            
            ax1.set_xlim(ax0.get_xlim())
            ax2.set_xlim(ax0.get_xlim())
            
            plt.tight_layout()
            plt.show()

    def add_behavior_plot(self, behavior):
        """Sets up plots for all nodes of the behavior."""
        print(behavior.name)
        self.plot_node(behavior.node_prior_intention)
        self.plot_node(behavior.node_intention)
        for name in behavior.nodes_cos:
            self.plot_node(behavior.nodes_cos[name])
            self.plot_node(behavior.nodes_cos_memory[name])

class Plotter():
    def __init__(self, figsize=(18,3)):
        self.figsize = figsize

    def plot_input(self, input):
        """Create a plot of the spike pattern of the input over time."""
        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(input.name)
        plotRaster(input.spike_times)
        plt.xlabel('Time steps')
        plt.ylabel('Input index')
        plt.xlim(0, input.number_of_timesteps)
        plt.title('Source Spikes')
        plt.tight_layout()
        plt.show()

    def plot_node(self, name, node_probes):
        """Creates plots for all probes of the node."""
        if (self.probe_current == None or self.probe_voltage == None or self.probe_spikes == None):
            print("Error: Node does not have all necessary probes to plot.")
        else:
            fig = plt.figure(figsize=(18,10))
            fig.suptitle(self.name)

            ax0 = plt.subplot(3,1,1)
            self.probe_current.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Current')
            plt.title('Current')

            ax1 = plt.subplot(3,1,2)
            self.probe_voltage.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Voltage')
            plt.title('Voltage')

            ax2 = plt.subplot(3,1,3)
            self.probe_spikes.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Neuron index')
            plt.title('Spikes')
            
            ax1.set_xlim(ax0.get_xlim())
            ax2.set_xlim(ax0.get_xlim())
            
            plt.tight_layout()
            plt.show()

    def plot_behavior(self):
        """Sets up plots for all nodes of the behavior."""
        print(self.name)
        self.node_prior_intention.plot()
        self.node_intention.plot()
        for name in self.nodes_cos:
            self.nodes_cos[name].plot()
            self.nodes_cos_memory[name].plot()

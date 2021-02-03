import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.node import Node
from dft_loihi.inputs.simulated_input import HomogeneousPiecewiseStaticInput
from dft_loihi.dft.util import connect


# set up the network
net = nxsdk.net.net.NxNet()

neurons_per_node = 1
simulated_input = HomogeneousPiecewiseStaticInput("input",
                                                  net,
                                                  neurons_per_node)
simulated_input.add_spike_rate(0, 100)
simulated_input.add_spike_rate(3000, 100)
simulated_input.add_spike_rate(8000, 100)
simulated_input.add_spike_rate(3000, 100)
simulated_input.add_spike_rate(0, 100)
simulated_input.create()

node = Node("node",
            net,
            number_of_neurons=neurons_per_node,
            tau_voltage=2,
            tau_current=10,
            self_excitation=0.10)

connect(simulated_input, node, 0.1, mask="one-to-one")

# run the network
time_steps = 500
net.run(time_steps)
net.disconnect()

# plot results
plotter = Plotter()
plotter.add_input_plot(simulated_input)
plotter.add_node_plot(node)
plotter.plot()

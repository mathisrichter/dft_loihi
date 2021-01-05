import sys
sys.path.append("/home/mathis/uni/diss/code/python/dft_loihi")
sys.path.append("/home/mathis/uni/diss/code/python/nxsdk/0.9.9/nxsdk-0.9.9")

import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.node import Node
from dft_loihi.inputs.simulated_input import SimulatedInput
from dft_loihi.dft.util import connect


timesteps = 500
neurons_per_node = 1

net = nxsdk.net.net.NxNet()

### setup the network
simulated_input = SimulatedInput("input", net, neurons_per_node, timesteps)
simulated_input.add_input_phase_spike_rate(100, 0.0)
simulated_input.add_input_phase_spike_rate(100, 40)
simulated_input.add_input_phase_spike_rate(100, 50)
simulated_input.add_input_phase_spike_rate(100, 40)
simulated_input.add_input_phase_spike_rate(100, 0.0)
simulated_input.create_input()

node = Node("node",
            net,
            number_of_neurons=neurons_per_node,
            tau_voltage=2,
            tau_current=10,
            self_excitation=0.08)

connect(simulated_input, node, 0.5, mask="one-to-one")

### run the network
net.run(timesteps)
net.disconnect()

### plot results
plotter = Plotter()
plotter.add_input_plot(simulated_input)
plotter.add_field_plot(node)
plotter.plot()
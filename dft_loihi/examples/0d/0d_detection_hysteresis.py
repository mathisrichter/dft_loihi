import os
import sys
sys.path.append("/home/mathis/uni/diss/code/python/dft_loihi")
sys.path.append("/home/mathis/uni/diss/code/python/nxsdk/0.9.9/nxsdk-0.9.9")

import nxsdk
import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.node import Node
from dft_loihi.inputs.simulated_input import SimulatedInput
from dft_loihi.dft.util import connect


timesteps = 1000
neurons_per_node = 5

net = nxsdk.net.net.NxNet()

### setup the network
simulated_input = SimulatedInput("input", net, neurons_per_node, timesteps)
simulated_input.add_input_phase_probability(200, 0.0)
simulated_input.add_input_phase_probability(200, 0.05)
simulated_input.add_input_phase_probability(200, 0.3)
simulated_input.add_input_phase_probability(200, 0.05)
simulated_input.add_input_phase_probability(200, 0.0)
simulated_input.create_input()

node = Node("node",
            net,
            number_of_neurons=neurons_per_node,
            tau_voltage=8,
            tau_current=10,
            self_excitation=0.03)

connect(simulated_input, node, 0.1, pattern="one-to-one")

### run the network
net.run(timesteps)
net.disconnect()

### plot results
plotter = Plotter()
plotter.add_input_plot(simulated_input)
plotter.add_node_plot(node)
plotter.plot()

import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.node import Node
from dft_loihi.inputs.simulated_input import SimulatedInput
from dft_loihi.dft.util import connect


timesteps = 100
neurons_per_node = 1

net = nxsdk.net.net.NxNet()

### setup the network
simulated_input = SimulatedInput("input", net, neurons_per_node, timesteps)
simulated_input.add_input_phase_probability(25, 0.0)
simulated_input.add_input_phase_probability(25, 0.5)
simulated_input.add_input_phase_probability(25, 0.1)
simulated_input.add_input_phase_probability(25, 0.0)
simulated_input.create_input()

node = Node("node", net, self_excitation=1.1)

connect(simulated_input, node, 1.1)

### run the network
net.run(timesteps)
net.disconnect()

### plot results
plotter = Plotter()
plotter.add_input_plot(simulated_input)
plotter.add_node_plot(node)
plotter.plot()

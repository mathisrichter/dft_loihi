import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.field import Field
from dft_loihi.dft.node import Node
from dft_loihi.dft.kernel import MultiPeakKernel
from dft_loihi.inputs.simulated_input import HomogeneousPiecewiseStaticInput,\
                                             GaussPiecewiseStaticInput
from dft_loihi.dft.util import connect


# set up the network
net = nxsdk.net.net.NxNet()

neurons_per_node = 1
homogeneous_input = HomogeneousPiecewiseStaticInput("input to node",
                                                    net,
                                                    shape=neurons_per_node)
homogeneous_input.add_spike_rate(0, 200)
homogeneous_input.add_spike_rate(1000, 200)
homogeneous_input.add_spike_rate(0, 100)
homogeneous_input.create()

node = Node("node",
            net,
            number_of_neurons=neurons_per_node,
            tau_voltage=2,
            tau_current=10,
            self_excitation=0.08)

connect(homogeneous_input, node, 0.5, mask="one-to-one")


field_domain = [-5, 5]
field_shape = 15
gauss_input = GaussPiecewiseStaticInput("input to field",
                                        net,
                                        domain=field_domain,
                                        shape=field_shape)
gauss_input.add_spike_rate(0, 2.5, 1.5, 100)
gauss_input.add_spike_rate(800, 2.5, 1.5, 300)
gauss_input.add_spike_rate(0, 2.5, 1.5, 100)
gauss_input.create()

kernel = MultiPeakKernel(amp_exc=0.47,
                         width_exc=2.5,
                         amp_inh=-0.35,
                         width_inh=3.85,
                         border_type="zeros")

field = Field("field",
              net,
              domain=field_domain,
              shape=field_shape,
              kernel=kernel,
              tau_voltage=2,
              tau_current=10,
              delay=3)

connect(gauss_input, field, 0.06, mask="one-to-one")
connect(node, field, 0.04, mask="full")

# run the network
time_steps = 500
net.run(time_steps)
net.disconnect()

# plot results
plotter = Plotter()
plotter.add_input_plot(homogeneous_input)
plotter.add_input_plot(gauss_input)
plotter.add_field_plot(node)
plotter.add_field_plot(field)
plotter.plot()

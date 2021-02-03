import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.field import Field
from dft_loihi.dft.kernel import MultiPeakKernel
from dft_loihi.inputs.simulated_input import GaussPiecewiseStaticInput
from dft_loihi.dft.util import connect


# set up the network
net = nxsdk.net.net.NxNet()

field_domain = [-5, 5]
field_shape = 30

gauss_input1 = GaussPiecewiseStaticInput("input 1", net, domain=field_domain, shape=field_shape)
gauss_input1.add_spike_rate(0, -0.775, 0.1, 100)
gauss_input1.add_spike_rate(11200, -0.775, 0.1, 400)
gauss_input1.create()

gauss_input2 = GaussPiecewiseStaticInput("input 2", net, domain=field_domain, shape=field_shape)
gauss_input2.add_spike_rate(0, 0.775, 0.1, 100)
gauss_input2.add_spike_rate(11200, 0.775, 0.1, 400)
gauss_input2.create()

kernel = MultiPeakKernel(amp_exc=0.40,
                         width_exc=2.0,
                         amp_inh=-0.28,
                         width_inh=4.0,
                         border_type="zeros")

field1 = Field("field",
               net,
               domain=field_domain,
               shape=field_shape,
               kernel=kernel,
               tau_voltage=2,
               tau_current=10,
               delay=3)

connect(gauss_input1, field1, 0.1, mask="one-to-one")
connect(gauss_input2, field1, 0.1, mask="one-to-one")

field2 = Field("field without kernel",
               net,
               domain=field_domain,
               shape=field_shape,
               kernel=None,
               tau_voltage=2,
               tau_current=10,
               delay=3)

connect(gauss_input1, field2, 0.1, mask="one-to-one")
connect(gauss_input2, field2, 0.1, mask="one-to-one")

# run the network
time_steps = 500
net.run(time_steps)
net.disconnect()

# plot results
plotter = Plotter()
plotter.add_input_plot(gauss_input1)
plotter.add_input_plot(gauss_input2)
plotter.add_field_plot(field1)
plotter.add_field_plot(field2)
plotter.plot()

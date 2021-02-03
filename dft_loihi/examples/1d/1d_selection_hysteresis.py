"""In this example, a peak initially forms either on the position of
input 1 (position 2.5) or at the position of input 2 (position -2.5).
This is a random decision and for the example to work, the initial
peak has to form at the position of input 1 (position 2.5).
Once that input ceases (after 100 time steps), the peak will decay and
switch to the position of input 2. It will stay there until that input
ceases at time step 300, even though input 1 was reintroduced at time
step 200. In the end, the peak will switch again.
The example shows selection hysteresis since the state of the field
(and thus its activation history) has an influence on the selection
decision."""

import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.field import Field
from dft_loihi.dft.kernel import SelectiveKernel
from dft_loihi.inputs.simulated_input import GaussPiecewiseStaticInput
from dft_loihi.dft.util import connect


# set up the network
net = nxsdk.net.net.NxNet()

field_domain = [-5, 5]
field_shape = 15

gauss_input1 = GaussPiecewiseStaticInput("input 1", net, domain=field_domain, shape=field_shape)
gauss_input1.add_spike_rate(11200, -2.5, 1.5, 300)
gauss_input1.add_spike_rate(0, -2.5, 1.5, 100)
gauss_input1.add_spike_rate(11200, -2.5, 1.5, 100)
gauss_input1.create()

gauss_input2 = GaussPiecewiseStaticInput("input 2", net, domain=field_domain, shape=field_shape)
gauss_input2.add_spike_rate(11200, 2.5, 1.5, 100)
gauss_input2.add_spike_rate(0, 2.5, 1.5, 100)
gauss_input2.add_spike_rate(11200, 2.5, 1.5, 300)
gauss_input2.create()

kernel = SelectiveKernel(amp_exc=0.38,
                         width_exc=2.5,
                         global_inh=0.285,
                         border_type="inhibition")

field = Field("field",
              net,
              domain=field_domain,
              shape=field_shape,
              kernel=kernel,
              tau_voltage=2,
              tau_current=10,
              delay=3)

connect(gauss_input1, field, 0.1, mask="one-to-one")
connect(gauss_input2, field, 0.1, mask="one-to-one")

# run the network
time_steps = 500
net.run(time_steps)
net.disconnect()

# plot results
plotter = Plotter()
plotter.add_input_plot(gauss_input1)
plotter.add_input_plot(gauss_input2)
plotter.add_field_plot(field)
plotter.plot()

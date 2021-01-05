import sys
import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.field import Field
from dft_loihi.dft.kernel import MultiPeakKernel
from dft_loihi.inputs.simulated_input import PiecewiseStaticInput
from dft_loihi.dft.util import connect

sys.path.append("/home/mathis/uni/diss/code/python/dft_loihi")
sys.path.append("/home/mathis/uni/diss/code/python/nxsdk/0.9.9/nxsdk-0.9.9")


timesteps = 500
net = nxsdk.net.net.NxNet()

# setup the network
field_domain = [-5, 5]
field_shape = 15

gauss_input1 = PiecewiseStaticInput("input 1", net, domain=field_domain, shape=field_shape)
gauss_input1.add_gaussian_spike_rate(0, -2.5, 1.5, 100)
gauss_input1.add_gaussian_spike_rate(800, -2.5, 1.5, 100)
gauss_input1.add_gaussian_spike_rate(11200, -2.5, 1.5, 100)
gauss_input1.add_gaussian_spike_rate(800, -2.5, 1.5, 100)
gauss_input1.add_gaussian_spike_rate(0, -2.5, 1.5, 100)
gauss_input1.create()

gauss_input2 = PiecewiseStaticInput("input 2", net, domain=field_domain, shape=field_shape)
gauss_input2.add_gaussian_spike_rate(0, 2.5, 1.5, 100)
gauss_input2.add_gaussian_spike_rate(800, 2.5, 1.5, 100)
gauss_input2.add_gaussian_spike_rate(11200, 2.5, 1.5, 100)
gauss_input2.add_gaussian_spike_rate(800, 2.5, 1.5, 100)
gauss_input2.add_gaussian_spike_rate(0, 2.5, 1.5, 100)
gauss_input2.create()

kernel = MultiPeakKernel(amp_exc=0.484,
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

connect(gauss_input1, field, 0.1, mask="one-to-one")
connect(gauss_input2, field, 0.1, mask="one-to-one")

# run the network
net.run(timesteps)
net.disconnect()

# plot results
plotter = Plotter()
plotter.add_input_plot(gauss_input1)
plotter.add_input_plot(gauss_input2)
plotter.add_field_plot(field)
plotter.plot()

import sys
import numpy as np
import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.field import Field
from dft_loihi.dft.kernel import SelectiveKernel
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
for center in np.linspace(-2.5, 2.5, 30):
    gauss_input1.add_gaussian_spike_rate(11200, center, 1.5, 10)
gauss_input1.add_gaussian_spike_rate(0, -2.5, 1.5, 100)
gauss_input1.create()

kernel = SelectiveKernel(amp_exc=0.485,
                         width_exc=2.5,
                         global_inh=0.2,
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

# run the network
net.run(timesteps)
net.disconnect()

# plot results
plotter = Plotter()
plotter.add_input_plot(gauss_input1)
plotter.add_field_plot(field)
plotter.plot()

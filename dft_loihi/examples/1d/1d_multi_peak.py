import sys
sys.path.append("/home/mathis/uni/diss/code/python/dft_loihi")
sys.path.append("/home/mathis/uni/diss/code/python/nxsdk/0.9.9/nxsdk-0.9.9")

import nxsdk.net.net

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.field import Field
from dft_loihi.inputs.simulated_input import GaussInput
from dft_loihi.dft.util import connect

timesteps = 500

net = nxsdk.net.net.NxNet()

### setup the network
field_shape = (10)

gauss_input = GaussInput("input", net, domain=[-5, 5], shape=field_shape)
gauss_input.add_input_phase(60, 4.5, 1.2, 100)
gauss_input.add_input_phase(1600, -4.5, 1.2, 100)
gauss_input.add_input_phase(1600, 3.5, 4.2, 100)
gauss_input.add_input_phase(1600, -3.5, 4.2, 100)
gauss_input.add_input_phase(60, -4.5, 1.2, 100)
gauss_input.create_input()

field = Field("field",
              net,
              sizes=field_shape,
              tau_voltage=2,
              tau_current=10)

connect(gauss_input, field, 0.5, pattern="one-to-one")

### run the network
net.run(timesteps)
net.disconnect()

### plot results
plotter = Plotter()
plotter.add_input_plot(gauss_input)
plotter.add_node_plot(field)
plotter.plot()

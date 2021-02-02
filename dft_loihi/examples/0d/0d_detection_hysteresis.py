from lava.core.generic.process import Process
from lava.core.generic.enums import Backend
from dft_loihi.dft.node import Node


class Architecture(Process):
    def _build(self, **kwargs):
        self.neurons_per_node = kwargs.pop("neurons_per_node", 1)
        self.time_steps = kwargs.pop("time_steps")

        self.node = Node(node_name="node",
                         number_of_neurons=self.neurons_per_node,
                         tau_voltage=2,
                         tau_current=10,
                         self_excitation=100)

        self.build_probes()

        return kwargs

    def build_probes(self):
        self.node.build_probes(buffer_size=self.time_steps)

    def plot(self):
        self.node.plot()


if __name__ == "__main__":
    time_steps = 200
    net = Architecture(neurons_per_node=1, time_steps=time_steps)
    net.run(time_steps, backend=Backend.TF)
    net.plot()

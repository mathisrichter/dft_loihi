import matplotlib.pyplot as plt
from lava.processes.n3.compartments import Compartments
from lava.core.generic.process import Process
from lava.core.generic.variable import InVar, OutVar
from dft_loihi.dft.util import tau_to_decay
from dft_loihi.dft.util import connect


class Node(Process):
    """A population of neurons that acts as a single dynamical node."""
    def _build(self, **kwargs):
        self.node_name = kwargs.pop("node_name")
        self.size = kwargs.pop("number_of_neurons", 1)
        tau_voltage = kwargs.pop("tau_voltage", 1)
        tau_current = kwargs.pop("tau_current", 1)
        threshold = kwargs.pop("threshold", 100)
        self_excitation = kwargs.pop("self_excitation", 0.0)
        bias_mant = kwargs.pop("bias_mant", 1000)
        bias_exp = kwargs.pop("bias_exp", 2)

        self.neurons = Compartments(
            name=self.node_name,
            num_compartments=self.size,
            voltage_decay=tau_to_decay(tau_voltage),
            current_decay=tau_to_decay(tau_current),
            refractory_delay=1,
            threshold=threshold,
            bias_mant=bias_mant,
            bias_exp=bias_exp)

        self.a_in = InVar.from_var(self.neurons.a_in)
        self.s_out = OutVar.from_var(self.neurons.s_out)

        if self_excitation > 0:
            connect(self, self, self_excitation, mask="full")

        return kwargs

    def build_probes(self, buffer_size):
        self.probe_current = self.neurons.probe('current', [0], buffer_size=buffer_size)
        self.probe_voltage = self.neurons.probe('voltage', [0], buffer_size=buffer_size)
        self.probe_spikes = self.neurons.probe('s_out', [0], buffer_size=buffer_size)

    def plot(self):
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(self.node_name)

        ax0 = plt.subplot(3, 1, 1)
        self.probe_current.plot()
        plt.xlabel('Time steps')
        plt.ylabel('Current')
        plt.title('Current')

        ax1 = plt.subplot(3, 1, 2)
        self.probe_voltage.plot()
        plt.xlabel('Time steps')
        plt.ylabel('Voltage')
        plt.title('Voltage')

        ax2 = plt.subplot(3, 1, 3)
        self.probe_spikes.plot()
        plt.xlabel('Time steps')
        plt.ylabel('Neuron index')
        plt.title('Spikes')
        plt.ylim(0, self.size)

        ax1.set_xlim(ax0.get_xlim())
        ax2.set_xlim(ax0.get_xlim())

        plt.tight_layout()
        plt.show()

import nengo
from nengo.networks.workingmemory import InputGatedMemory as WM
from nengo.spa.module import Module

diff_gain = 15
n_neurons = 100

class IncNet(Module):
    def __init__(self, d, inc_vocab, label=None, seed=None, add_to_container=None):
        super(IncNet, self).__init__(label, seed, add_to_container)

        with self:
            self.mem1 = WM(n_neurons, d, difference_gain=diff_gain)
            self.mem1.label = "mem1"
            self.mem2 = WM(n_neurons, d, difference_gain=diff_gain)
            self.mem2.label = "mem2"

            nengo.Connection(self.mem1.output, self.mem2.input)
            nengo.Connection(
                self.mem2.output, self.mem1.input,
                transform=inc_vocab['ONE'].get_convolution_matrix())

        self.inputs = dict(default=(self.mem1.input, inc_vocab), gate1=(self.mem1.gate, inc_vocab), gate2=(self.mem1.gate, inc_vocab),)
        self.outputs = dict(default=(self.mem1.output, inc_vocab))
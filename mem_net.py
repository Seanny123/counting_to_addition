import nengo
from nengo.networks.workingmemory import InputGatedMemory as WM
from nengo.spa.module import Module

class MemNet(Module):
    def __init__(self, d, mem_vocab, label=None, seed=None, add_to_container=None):
        super(IncNet, self).__init__(label, seed, add_to_container)

        gate_vocab = spa.Vocabulary(16)
        gate_vocab.parse("OPEN+CLOSE")

        with self:
            self.mem = WM(100, d, difference_gain=15)
            self.mem.label = "mem"
            self.gate = nengo.Node(d)

            nengo.Connection(self.gate, self.mem.gate,
                             transform=np.array([vocab.parse("OPEN").v]))

        self.inputs = dict(default=(self.mem.input, mem_vocab), gate=(self.gate, gate_vocab))
        self.outputs = dict(default=(self.mem.output, mem_vocab))
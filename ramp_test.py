import nengo
from nengo import spa

import numpy as np

from nengo.spa.module import Module
from spaun_config import SpaunConfig

cfg = SpaunConfig()


class RampNet(Module):
    def __init__(self, d, ramp_vocab, label=None, seed=None, add_to_container=None):
        super(RampNet, self).__init__(label, seed, add_to_container)

        with self:
            self.ramp_integrator = nengo.Ensemble(cfg.n_neurons_cconv, 1, radius=1.1)
            self.start = nengo.Node(size_in=d)
            self.reset = nengo.Node(size_in=d)

            # 0.3 to peak is good enough for me, but it might be taking off on it's own now...
            print(cfg.mtr_ramp_synapse * cfg.mtr_ramp_scale)
            nengo.Connection(self.start, self.ramp_integrator,
                             transform=np.array([ramp_vocab.parse("START").v * 0.2]),
                             synapse=cfg.mtr_ramp_synapse)
            nengo.Connection(self.ramp_integrator, self.ramp_integrator,
                             transform=1.0,
                             synapse=cfg.mtr_ramp_synapse)
            nengo.Connection(self.reset, self.ramp_integrator.neurons,
                             transform= np.array([ramp_vocab.parse("RESET").v * -3] * cfg.n_neurons_cconv))

            self.inputs = dict(start=(self.start, ramp_vocab), reset=(self.reset, ramp_vocab),)
            self.outputs = dict(default=(self.ramp_integrator, ramp_vocab))

D = 16
vocab = spa.Vocabulary(D)
vocab.parse("START+RESET")

model = spa.SPA(vocabs=[vocab])

with model:
    model.reset_in = spa.State(D)
    model.start_in = spa.State(D)

    model.ramp = RampNet(D, vocab)

    actions = spa.Actions(
        "ramp_start = start_in",
        "ramp_reset = reset_in"
    )

    model.cortical = spa.Cortical(actions)

    def reset_func(t):
        if(t < 0.6):
            return '0'
        elif(t < 0.5):
            return "RESET"
        else:
            return '0'

    def start_func(t):
        if(t < 0.06):
            return "START"
        else:
            return "0"

    #model.input = spa.Input(reset_in=reset_func)#, start_in=start_func)
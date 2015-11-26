import nengo
from nengo.spa.module import Module
from spaun_config import SpaunConfig

cfg = SpaunConfig()


class IncNet(Module):
    def __init__(self, d, ramp_vocab, label=None, seed=None, add_to_container=None):
        super(IncNet, self).__init__(label, seed, add_to_container)

        with self:
            self.ramp_integrator = nengo.Ensemble(cfg.n_neurons_cconv, 1, radius=1.1)
            self.start = nengo.Node(size_in=d)
            self.reset = nengo.Node(size_in=d)

            nengo.Connection(self.start, self.ramp_integrator,
                             transform=np.array([ramp_vocab.parse("START").v]) * cfg.mtr_ramp_synapse * cfg.mtr_ramp_scale,
                             synapse=cfg.mtr_ramp_synapse)
            nengo.Connection(self.ramp_integrator, self.ramp_integrator,
                             synapse=cfg.mtr_ramp_synapse)
            nengo.Connection(self.reset, self.ramp_integrator.neurons,
                             transform= np.array([vocab.parse("RESET").v]) -3 * cfg.n_neurons_cconv)

            self.inputs = dict(start=(self.start, ramp_vocab), reset=(self.reset, ramp_vocab),)
            self.outputs = dict(default=(self.ramp_integrator, ramp_vocab))
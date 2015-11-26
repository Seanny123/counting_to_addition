import nengo
from nengo import spa
from spaun_config import SpaunConfig


cfg = SpaunConfig()

model = spa.SPA(label="ramp")

with model:
    ramp_integrator = nengo.Ensemble(cfg.n_neurons_cconv, 1, radius=1.1)
    motor_go = nengo.Node([0])
    reset = nengo.Node([0])

    nengo.Connection(motor_go, ramp_integrator,
                     transform=cfg.mtr_ramp_synapse * cfg.mtr_ramp_scale,
                     synapse=cfg.mtr_ramp_synapse)
    nengo.Connection(ramp_integrator, ramp_integrator,
                     synapse=cfg.mtr_ramp_synapse)
    nengo.Connection(reset, ramp_integrator.neurons,
                     transform=[[-3]] * cfg.n_neurons_cconv)

import nengo
from nengo import spa
from spaun_config import SpaunConfig


cfg = SpaunConfig()
D = 16
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

    model.relay = spa.State(1)

    model.in_state = spa.State(D)

    c_act = spa.Actions("relay = 1")
    model.cortical = spa.Cortical(c_act)

    """
    actions = spa.Actions()

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
    """

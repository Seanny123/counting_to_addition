import nengo
import spaun_config

ramp_integrator = nengo.Ensemble(cfg.n_neurons_cconv, 1, radius=1.1)
ramp_reset_thresh = cfg.make_thresh_ens_net(0.91, radius=1.1)
ramp_reset_hold = cfg.make_thresh_ens_net(0.07,
                                               thresh_func=lambda x: x)
ramp_75 = cfg.make_thresh_ens_net(0.75)
ramp_50_75 = cfg.make_thresh_ens_net(0.5)

nengo.Connection(self.motor_go, ramp_integrator,
                 transform=cfg.mtr_ramp_synapse * cfg.mtr_ramp_scale,
                 synapse=cfg.mtr_ramp_synapse)
nengo.Connection(ramp_integrator, ramp_integrator,
                 synapse=cfg.mtr_ramp_synapse)

nengo.Connection(ramp_integrator, ramp_reset_thresh.input)
nengo.Connection(ramp_integrator, ramp_75.input)
nengo.Connection(ramp_integrator, ramp_50_75.input)
nengo.Connection(ramp_75.output, ramp_50_75.input, transform=-3)

nengo.Connection(self.motor_init.output, ramp_reset_hold.input,
                 transform=1.75, synapse=0.015)
nengo.Connection(ramp_reset_thresh.output, ramp_reset_hold.input,
                 transform=1.75, synapse=0.015)
nengo.Connection(ramp_reset_hold.output,
                 ramp_reset_hold.input,
                 transform=cfg.mtr_ramp_reset_hold_transform)
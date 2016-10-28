import numpy as np
import nengo

# Note that D is equal to the dimensions of the addend
from constants import *
from ntmdists import ScatteredHypersphere


def encoders(pointers, k, rng):
    m, d = pointers.shape
    dist = ScatteredHypersphere(surface=True)
    print(d)
    return dist.sample(m*k, d, rng=rng).reshape(m, k, d)


class NullSolver(nengo.solvers.Lstsq):
    """Zero decoder solver."""

    def __call__(self, A, Y, rng=None, E=None):
        return np.zeros((A.shape[1], Y.shape[1])), {}


def build_hetero_mem(in_d, out_d, encoders, intercept, pes_rate=0.01, pes_tau=1e-16, voja_rate=0.005, voja_tau=None):
    """Heteroassociative memory builder"""

    m, k, d = encoders.shape

    # Memory should be implemented as a single m * k neuron population
    # Aaron says otherwise is okay too. If anything, more neurons will probably help
    n = m * k

    # Reshape encoders to fit this single population
    ens_encoders = np.reshape(encoders, (n, d))

    with nengo.Network(seed=SEED) as net:
        net.input = nengo.Node(size_in=in_d, label="input")
        net.output = nengo.Node(size_in=out_d, label="output")

        # The ensemble where the actual learning happens
        net.ens = nengo.Ensemble(
            n, d, encoders=ens_encoders, intercepts=[intercept]*n,
            eval_points=[ens_encoders[i] for i in range(0, n, k)],
            label="ens")
        net.voja_rule = nengo.Voja(voja_tau, learning_rate=voja_rate)
        net.in_conn = nengo.Connection(
            net.input, net.ens, synapse=None,
            learning_rule_type=net.voja_rule)

        pes_rule = nengo.PES(learning_rate=pes_rate, pre_tau=pes_tau)
        net.out_conn = nengo.Connection(
            net.ens, net.output, function=lambda x: np.zeros(out_d),
            solver=NullSolver(), synapse=None,
            learning_rule_type=pes_rule)

    return net


def rebuild_hetero_mem(in_w, out_w, prev_ens, pes_rate=0.01, pes_tau=1e-16, voja_rate=0.005, voja_tau=None):
    """Heteroassociative memory builder"""

    with nengo.Network(seed=SEED) as net:
        net.input = nengo.Node(size_in=prev_ens.dimensions, label="input")
        net.output = nengo.Node(size_in=prev_ens.dimensions / 2, label="output")

        # The ensemble where the actual learning happens
        ens = nengo.Ensemble(
            prev_ens.n_neurons, prev_ens.dimensions,
            encoders=in_w, intercepts=prev_ens.intercepts,
            n_eval_points=0, label="ens")
        net.in_conn = nengo.Connection(
            net.input, ens, synapse=None,
            learning_rule_type=nengo.Voja(voja_tau, learning_rate=voja_rate))

        pes_rule = nengo.PES(learning_rate=pes_rate, pre_tau=pes_tau)
        net.out_conn = nengo.Connection(
            ens.neurons, net.output, transform=out_w, synapse=None,
            learning_rule_type=pes_rule)

    return net

# no voja, single concatenated vector input
# used for making plot showing decoders continually switching back and forth

# make sure it messes up in the way you expect
# get the decoder data
# try to show the decoder values changing
# - look for decoders who's values change with every question

from utils import gen_env_list, gen_vocab
from hetero_mem import build_hetero_mem, encoders
from constants import n_neurons, D, dt

import nengo
from nengo import spa

import numpy as np

plot_res = False

# Learning rates
pes_rate = 0.01
voja_rate = 0.005

## Generate the vocab
rng = np.random.RandomState(0)
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}

max_sum = 5
max_num = max_sum - 2

number_list, vocab = gen_vocab(number_dict, max_num, D, rng)

join_num = "+".join(number_list[0:max_num])

## Create inputs and expected outputs
q_list, q_norm_list, ans_list = gen_env_list(number_dict, number_list, vocab, max_sum)

num_items = len(q_list)
period = 0.3
repeats = 5
T = period * num_items * (1 + repeats)
shuffled = range(0, len(q_list))
rng.shuffle(shuffled)
print(shuffled)


def shuffle_func(t):
    if round((t % period * num_items) * 1000) == 0:
        rng.shuffle(shuffled)

    return shuffled


def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period/dt))
    if i_every != period/dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def f(t):
        i = int(round((t - dt)/dt))  # t starts at dt
        return x[shuffled[(i/i_every) % len(x)]]

    return f


with spa.SPA(vocabs=[vocab], label="Fast Net", seed=0) as model:
    env_keys = nengo.Node(cycle_array(q_norm_list, period, dt))
    env_values = nengo.Node(cycle_array(ans_list, period, dt))
    shuffle_node = nengo.Node(shuffle_func)

    recall = nengo.Node(size_in=D)
    learning = nengo.Node(output=lambda t: -int(t >= (T-period*num_items)))

    ## Generate hetero mem
    K = 400
    # This is usually calculated
    c = 0.51
    e = encoders(np.array(q_norm_list), K, rng)
    het_mem = build_hetero_mem(D*2, D, e, c, pes_rate=pes_rate, voja_rate=voja_rate)

    nengo.Connection(env_keys, het_mem.input, synapse=None)
    nengo.Connection(het_mem.output, recall)

    # Create the error population
    error = nengo.Ensemble(n_neurons*8, D)
    nengo.Connection(learning, error.neurons, transform=[[10.0]]*n_neurons*8,
                     synapse=None)
    nengo.Connection(learning, het_mem.in_conn.learning_rule, synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(env_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, het_mem.out_conn.learning_rule)

    # Setup probes
    p_keys = nengo.Probe(env_keys, synapse=None, sample_every=0.01)
    p_values = nengo.Probe(env_values, synapse=None, sample_every=0.01)
    p_learning = nengo.Probe(learning, synapse=None, sample_every=0.01)
    p_error = nengo.Probe(error, synapse=0.01)
    p_recall = nengo.Probe(recall, synapse=None, sample_every=0.01)
    p_weights = nengo.Probe(het_mem.out_conn, 'weights', synapse=None, sample_every=0.01)

sim = nengo.Simulator(model, dt=dt)
sim.run(T)

t = sim.trange()

if plot_res:
    import matplotlib.pyplot as plt

    # figure out how to put these into a subplot
    plt.figure()
    plt.title("Error")
    plt.plot(t, np.linalg.norm(sim.data[p_error], axis=1))

    plt.figure()
    plt.title("Keys_1")
    plt.plot(t, spa.similarity(sim.data[p_keys][:, :D], vocab))
    plt.legend(vocab.keys, loc='best')

    plt.figure()
    plt.title("Keys_2")
    plt.plot(t, spa.similarity(sim.data[p_keys][:, D:], vocab))
    plt.legend(vocab.keys, loc='best')

    plt.figure()
    plt.title("Result")
    plt.plot(t, spa.similarity(sim.data[p_recall], vocab))
    plt.legend(vocab.keys, loc='best')
    plt.ylim(-1.5, 1.5)

    plt.figure()
    plt.title("Actual Answer")
    plt.plot(t, spa.similarity(sim.data[p_values], vocab))
    plt.legend(vocab.keys, loc='best')
    plt.ylim(-1.5, 1.5)

    plt.show()

# I should make a wrapper for doing this quickly
base_name = "hetmem"
np.savez_compressed("data/%s_learning_data" % base_name, p_keys=sim.data[p_keys], p_recall=sim.data[p_recall],
                    p_error=sim.data[p_error], p_weights=sim.data[p_weights], p_values=sim.data[p_values])
np.savez_compressed("data/%s_learning_vocab" % base_name, keys=vocab.keys, vecs=vocab.vectors)

# no voja, single concatenated vector input
# used for making plot showing decoders continually switching back and forth

# make sure it messes up in the way you expect
# get the decoder data
# try to show the decoder values changing
# - look for decoders who's values change with every question

import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools
from random import shuffle
import ipdb

## Generate the vocab
rng = np.random.RandomState(0)
D = 32
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}

# This should be set to 10 for the actual final test
max_sum = 9
max_num = max_sum - 2

number_list, vocab = gen_vocab(number_dict, max_num, D, rng)

join_num = "+".join(number_list[0:max_num])

## Create inputs and expected outputs
q_list, _, ans_list = gen_env_list(number_dict, number_list, vocab, max_sum)

num_items = len(q_list)
dt = 0.001
period = 0.3
T = period*num_items/dt
shuffled = range(0,len(q_list))
shuffle(shuffled)
print(shuffled)

def shuffle_func(t):
    if(round((t % period * num_items) * 1000) == 0):
        shuffle(shuffled)
    return shuffled

def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period/dt))
    if i_every != period/dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))
    def f(t):
        i = int(round((t - dt)/dt))  # t starts at dt
        return x[shuffled[(i/i_every)%len(x)]]
    return f

with model:
    env_keys = nengo.Node(cycle_array(q_list, period, dt))
    env_values = nengo.Node(cycle_array(ans_list, period, dt))
    shuffle_node = nengo.Node(shuffle_func)

    adder = nengo.Ensemble(n_neurons, D*2, radius=2.0)
    recall = nengo.Node(size_in=D)
    learning = nengo.Node(output=lambda t: -int(t>=(T-period)))

    nengo.Connection(env_keys, adder)
    conn_out = nengo.Connection(adder, recall, learning_rule_type=nengo.PES(1e-5),
                                function=lambda x: np.zeros(D))

    # Create the error population
    error = nengo.Ensemble(n_neurons, D)
    nengo.Connection(learning, error.neurons, transform=[[10.0]]*n_neurons,
                     synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(env_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Setup probes
    p_keys = nengo.Probe(env_keys, synapse=None)
    p_values = nengo.Probe(env_values, synapse=None)
    p_learning = nengo.Probe(learning, synapse=None)
    p_error = nengo.Probe(error, synapse=0.005)
    p_recall = nengo.Probe(recall, synapse=None)
    p_weights = nengo.Probe(conn_out, 'weights', synapse=None, sample_every=0.01)

sim = nengo.Simulator(model, dt=dt)
sim.run(T)

t = sim.trange()

import matplotlib.pyplot as plt

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
plt.title("Acutal Answer")
plt.plot(t, spa.similarity(sim.data[p_values], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.show()

ipdb.set_trace()
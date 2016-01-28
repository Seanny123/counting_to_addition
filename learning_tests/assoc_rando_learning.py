# same as assoc learning, but with no unitary vector constraint

import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools
from random import shuffle
import ipdb

D = 32
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 4
number_list = number_ordered.keys()
for i in range(number_range):
    print(number_list[i])
    vocab.parse(number_list[i])

model = spa.SPA(vocabs=[vocab], label="Count Net", seed=0)
n_neurons = 100

q_list = []
ans_list = []
for val in itertools.product(number_list, number_list):
    ans_val = number_dict[val[0]] + number_dict[val[1]]
    if ans_val <= number_range:
        q_list.append(
            np.concatenate(
                (vocab.parse(val[0]).v, vocab.parse(val[1]).v)
            )
        )
        ans_list.append(
            vocab.parse(number_list[ans_val-1]).v
        )
        print("%s+%s=%s" %(val[0], val[1], number_list[ans_val-1]))

num_items = len(q_list)
reps = 500
dt = 0.001
period = 0.4
T = period*num_items*reps
shuffled = range(0,len(q_list))

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
        if t % period < 0.3:
            return x[shuffled[(i/i_every)%len(x)]]
        else:
            return np.zeros(x[0].size)
    return f

def learning_func(t):
    if t % period < 0.3:
        return -int(t>=(T-period*num_items))
    else:
        return 0

with model:
    env_keys = nengo.Node(cycle_array(q_list, period, dt))
    env_values = nengo.Node(cycle_array(ans_list, period, dt))
    shuffle_node = nengo.Node(shuffle_func)

    # Learning with an associative memory provides dimension reduction
    adder = nengo.networks.AssociativeMemory(input_vectors=q_list, n_neurons=n_neurons)
    adder.add_wta_network()
    adder_out = nengo.Ensemble(n_neurons*8, len(ans_list))
    recall = nengo.Node(size_in=D)
    learning = nengo.Node(output=learning_func)

    nengo.Connection(env_keys, adder.input)
    nengo.Connection(adder.elem_output, adder_out)

    conn_out = nengo.Connection(adder_out, recall, learning_rule_type=nengo.PES(1e-5),
                                function=lambda x: np.zeros(D))

    # Create the error population
    error = nengo.Ensemble(n_neurons*8, D)
    nengo.Connection(learning, error.neurons, transform=[[10.0]]*n_neurons*8,
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


sim = nengo.Simulator(model, dt=dt)
sim.run(T)

t = sim.trange()

import matplotlib.pyplot as plt
"""
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

win = -400*18
plt.figure()
plt.title("Result")
plt.plot(t[win:], spa.similarity(sim.data[p_recall][win:,:], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Acutal Answer")
plt.plot(t[win:], spa.similarity(sim.data[p_values][win:,:], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Learning")
plt.plot(t[win:], sim.data[p_learning][win:,:])
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Keys_1")
plt.plot(t[win:], spa.similarity(sim.data[p_keys][win:, :D], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Keys_2")
plt.plot(t[win:], spa.similarity(sim.data[p_keys][win:, D:], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.plot(t, sim.data[p_dum])
plt.show()
plt.plot(t, sim.data[p_add])
plt.show()

plt.show()
"""
ipdb.set_trace()

np.savez_compressed("data/no_unitary_learn_fig_data", t=t, p_keys = sim.data[p_keys], p_values = sim.data[p_values], p_learning = sim.data[p_learning], p_error = sim.data[p_error], p_recall = sim.data[p_recall])

keys = af["p_keys"]
values = af["p_values"]
learning = af["p_learning"]
error = af["p_error"]
recall = af["p_recall"]
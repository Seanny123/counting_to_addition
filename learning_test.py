import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools

D = 64
less_D = 32
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
number_dict = {"ZERO":0, "ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 5
vocab.parse("ZERO")
number_list = number_ordered.keys()
for i in range(1, number_range):
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

join_num = "+".join(number_list[0:number_range])
num_ord_filt = OrderedDict(number_ordered.items()[:number_range])

print(join_num)

q_list = []
ans_list = []
for val in itertools.product(num_ord_filt, num_ord_filt):
    ans_val = num_ord_filt[val[0]] + num_ord_filt[val[1]]
    if ans_val < number_range:
        q_list.append(
            np.concatenate(
                (vocab.parse(val[0]).v, vocab.parse(val[1]).v)
            )
        )
        ans_list.append(
            vocab.parse(num_ord_filt.keys()[ans_val]).v
        )

num_items = len(q_list)
dt = 0.001
period = 0.3
T = period*num_items*2

intercept = (np.dot(np.array(q_list), np.array(q_list).T) - np.eye(num_items)).flatten().max()

def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period/dt))
    if i_every != period/dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))
    def f(t):
        i = int(round((t - dt)/dt))  # t starts at dt
        return x[(i/i_every)%len(x)]
    return f

def create_env():
    with nengo.Network(label="env") as env:
        env.keys = nengo.Node(cycle_array(q_list, period, dt))
        env.values = nengo.Node(cycle_array(ans_list, period, dt))
        env.learning = nengo.Node(output=lambda t: -int(t>=T/2))
    return env

num_vocab = vocab.create_subset(num_ord_filt.keys())

model = spa.SPA(vocabs=[vocab], label="Count Net", seed=0)
n_neurons = 500

with model:
    env = create_env()
    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)
    adder = nengo.Ensemble(n_neurons, D*2, intercepts=[intercept]*n_neurons)
    recall = nengo.Node(size_in=D)

    conn_in = nengo.Connection(env.keys, adder, synapse=None,
                           learning_rule_type=voja)
    nengo.Connection(env.learning, conn_in.learning_rule, synapse=None)

    conn_out = nengo.Connection(adder, recall, learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.zeros(D))

    # Create the error population
    error = nengo.Ensemble(n_neurons, D)
    nengo.Connection(env.learning, error.neurons, transform=[[10.0]]*n_neurons,
                     synapse=None)
    
    # Calculate the error and use it to drive the PES rule
    nengo.Connection(env.values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Probing for GUI
    model.in_1 = spa.State(D)
    model.in_2 = spa.State(D)

    model.out = spa.State(D)

    nengo.Connection(env.keys[D:], model.in_1.input)
    nengo.Connection(env.keys[:D], model.in_2.input)
    nengo.Connection(recall, model.out.input)

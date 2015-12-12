import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools
import ipdb

sim_mode = "gui"
D = 64
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 4
number_list = number_ordered.keys()
for i in range(number_range):
    print(number_list[i])
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

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
reps = 250
dt = 0.001
period = 0.3
T = period*num_items*reps
q_normed = np.array(q_list)/2
intercept = (np.dot(q_normed, q_normed.T) - np.eye(num_items)).flatten().max()
print("Intercept: %s" % intercept)

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

def create_env():
    with nengo.Network(label="env") as env:
        env.keys = nengo.Node(cycle_array(q_normed, period, dt))
        env.values = nengo.Node(cycle_array(ans_list, period, dt))
    return env

model = spa.SPA(vocabs=[vocab], label="Count Net", seed=0)
n_neurons = 800

with model:
    env = create_env()
    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)
    # TODO: Switch to Uniform Hypersphere with minimum later
    adder = nengo.Ensemble(n_neurons, D*2, intercepts=[intercept]*n_neurons)
    recall = nengo.Node(size_in=D)
    learning = nengo.Node(output=learning_func)

    conn_in = nengo.Connection(env.keys, adder, synapse=None,
                               learning_rule_type=voja)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)

    conn_out = nengo.Connection(adder, recall, learning_rule_type=nengo.PES(1e-5),
                                function=lambda x: np.zeros(D))

    # Create the error population
    error = nengo.Ensemble(n_neurons, D)
    nengo.Connection(learning, error.neurons, transform=[[10.0]]*n_neurons,
                     synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(env.values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    err_node = nengo.Node(lambda t, x: np.linalg.norm(x), size_in=D)
    nengo.Connection(error, err_node)

    # OH SHIT IT'S THE INTERCEPTS
    dummy = nengo.Ensemble(n_neurons, D*2)
    nengo.Connection(env.keys, dummy, synapse=None)


    # Probing modes
    if(sim_mode == "gui"):
        model.in_1 = spa.State(D)
        model.in_2 = spa.State(D)

        model.out = spa.State(D)
        model.ans = spa.State(D)

        nengo.Connection(env.keys[D:], model.in_1.input)
        nengo.Connection(env.keys[:D], model.in_2.input)
        nengo.Connection(recall, model.out.input)
        nengo.Connection(env.values, model.ans.input)

    elif(sim_mode == "cli"):
        p_keys = nengo.Probe(env.keys, synapse=None)
        p_learning = nengo.Probe(learning, synapse=None)
        p_recall = nengo.Probe(recall, synapse=None)
        p_values = nengo.Probe(env.values, synapse=None)
        p_err_mag = nengo.Probe(err_node, synapse=0.005)
        p_err = nengo.Probe(error, synapse=0.005)
        p_dum = nengo.Probe(dummy, synapse=0.005)
        p_add = nengo.Probe(adder, synapse=0.005)


if(sim_mode == "cli"):
    sim = nengo.Simulator(model, dt=dt)
    #sim.run(T)
    sim.run(period*num_items*2)
    import matplotlib.pyplot as plt

    t = sim.trange()
    ipdb.set_trace()

    plt.plot(t, sim.data[p_err])
    plt.show()

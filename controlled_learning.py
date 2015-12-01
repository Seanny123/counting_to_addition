import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools
import ipdb

sim_mode = "gui"
D = 32
less_D = 32
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng, max_similarity=1.0)
number_dict = {"ZERO":0, "ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 3
vocab.parse("ZERO")
number_list = number_ordered.keys()
for i in range(1, number_range):
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

join_num = "+".join(number_list[0:number_range])
num_ord_filt = OrderedDict(number_ordered.items()[:number_range])

print(join_num)

q_list = []
for val in itertools.product(num_ord_filt, num_ord_filt):
    ans_val = num_ord_filt[val[0]] + num_ord_filt[val[1]]
    if ans_val < number_range:
        q_list.append(
            np.concatenate(
                (vocab.parse(val[0]).v, vocab.parse(val[1]).v)
            )
        )

num_items = len(q_list)
dt = 0.001
period = 0.3
T = period*num_items*2
q_normed = np.array(q_list)
q_normed = q_normed/np.linalg.norm(q_normed)
intercept = (np.dot(q_normed, q_normed.T) - np.eye(num_items)).flatten().max()
print("Intercept: %s" % intercept)

num_vocab = vocab.create_subset(num_ord_filt.keys())

model = spa.SPA(vocabs=[vocab], label="Count Net", seed=0)
n_neurons = 500

with model:
    model.in_1 = spa.State(D)
    model.in_2 = spa.State(D)

    model.out = spa.State(D)
    model.ans = spa.State(D)

    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)
    adder = nengo.Ensemble(n_neurons, D*2, intercepts=[intercept]*n_neurons)
    recall = nengo.Node(size_in=D)
    comb_node = nengo.Node(size_in=2*D)

    conn_in = nengo.Connection(comb_node, adder, synapse=None,
                           learning_rule_type=voja)

    conn_out = nengo.Connection(adder, recall, learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.zeros(D))

    nengo.Connection(model.in_1.output, comb_node[D:])
    nengo.Connection(model.in_2.output, comb_node[:D])

    nengo.Connection(recall, model.out.input)

    # Create the error population
    error = nengo.Ensemble(n_neurons, D)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(model.ans.output, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    err_node = nengo.Node(lambda t, x: np.linalg.norm(x), size_in=D)
    nengo.Connection(error, err_node)

    # OH SHIT IT'S THE INTERCEPTS
    dummy = nengo.Ensemble(n_neurons, D*2, intercepts=[intercept]*n_neurons)
    nengo.Connection(comb_node, dummy, synapse=None)
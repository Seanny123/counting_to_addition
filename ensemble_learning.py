import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools
import ipdb

D = 32
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D)
number_dict = {"ZERO":0, "ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 4
number_list = number_ordered.keys()
for i in range(0, number_range):
    vocab.parse(number_list[i])

model = spa.SPA(vocabs=[vocab], label="Count Net", seed=0)
n_neurons = 800

with model:
    model.in_1 = spa.State(D)
    model.in_2 = spa.State(D)

    model.out = spa.State(D)
    model.ans = spa.State(D)

    # maybe switch this to an Ensemble Array?
    adder = nengo.Ensemble(n_neurons, D*2)
    model.assoc = spa.AssociativeMemory(input_vocab=vocab, wta_output=True, threshold_output=True)
    recall = nengo.Node(size_in=D)
    learning = nengo.Node([0])

    nengo.Connection(model.in_1.output, adder[D:])
    nengo.Connection(model.in_2.output, adder[:D])
    conn_out = nengo.Connection(adder, model.assoc.input, learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.zeros(D))
    nengo.Connection(model.assoc.output, recall)

    nengo.Connection(model.assoc.input, model.out.input)

    # Create the error population
    error = nengo.Ensemble(n_neurons, D)
    nengo.Connection(learning, error.neurons, transform=[[-10.0]]*n_neurons,
                     synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(model.ans.output, error, transform=-1, synapse=None)
    nengo.Connection(model.assoc.input, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    err_node = nengo.Node(lambda t, x: np.linalg.norm(x), size_in=D)
    nengo.Connection(error, err_node)
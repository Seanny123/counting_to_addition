import nengo
from nengo import spa
import numpy as np
import ipdb

from collections import OrderedDict

from inc_network import IncNet

D = 64
vocab = spa.Vocabulary(D, unitary=["ONE"])
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 4
vocab.parse("NONE")
vocab.parse("ONE")
number_list = number_ordered.keys()
for i in range(number_range):
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

join_num = "+".join(number_list[0:number_range])
num_ord_filt = OrderedDict(number_ordered.items()[:number_range])

print(join_num)

num_vocab = vocab.create_subset(num_ord_filt.keys()+["NONE"])
state_vocab = vocab.create_subset(["NONE", "CNT1", "CNT2"])

model = spa.SPA(vocabs=[vocab], label="Count Net")

with model:

    def gate_func(t):
        if(t%0.2 < 0.1):
            return [0, 1]
        else:
            return [1, 0]

    gate = nengo.Node(gate_func)
    model.input_state = spa.State(D, vocab=num_vocab)

    # fuck
    def res_func(t):
        if(t < 0.2):
            return "ONE"
        else:
            return "0"

    model.count_res = IncNet(D, num_vocab, label="result")
    model.input = spa.Input(input_state=res_func)

    # making these connections while the network was expanded caused an epic bug
    nengo.Connection(gate[0], model.count_res.mem1.gate)
    nengo.Connection(gate[1], model.count_res.mem2.gate)

    # TODO: add the dot transform
    cortical_actions = spa.Actions(
        "count_res = input_state"
    )

    model.cortical = spa.Cortical(cortical_actions)
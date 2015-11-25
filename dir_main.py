import nengo
from nengo import spa
import numpy as np
import ipdb

from collections import OrderedDict

from inc_network import IncNet

D = 128
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 5
vocab.parse("NONE")
vocab.parse("ONE")
number_list = number_ordered.keys()
for i in range(number_range):
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

join_num = "+".join(number_list[0:number_range])
print(join_num)
num_ord_filt = OrderedDict(number_ordered.items()[:number_range])

num_vocab = vocab.create_subset(num_ord_filt.keys()+["NONE"])
state_vocab = vocab.create_subset(["NONE", "CNT1", "CNT2"])

model = spa.SPA(vocabs=[vocab], label="Count Net", seed=0)

with model:
    model.q1 = spa.State(D, vocab=num_vocab)
    model.q2 = spa.State(D, vocab=num_vocab)

    model.op_state = spa.State(D, vocab=state_vocab)

    model.count_res = IncNet(D, num_vocab, label="result")
    model.count_tot = IncNet(D, num_vocab, label="total")

    model.count_fin = spa.State(D, vocab=num_vocab, feedback=1)

    step = 0.1
    def q1_func(t):
        if(t < step*2):
            return "TWO"
        else:
            return "0"

    def q2_func(t):
        if(t < step):
            return "TWO"
        else:
            return "0"

    def op_state_func(t):
        if(t < step):
            return "NONE"
        elif(t < step*2):
            return "CNT2"
        elif(t < step*3):
            return "CNT1"
        else:
            return "0"

    def total_func(t):
        if(t < step):
            return "ONE"
        else:
            return "0"

    model.input = spa.Input(q1=q1_func, q2=q2_func, op_state=op_state_func, count_tot=total_func)


    # So... Apparently, the gates aren't supposed to go to -1...
    # Maybe I should add a thresholding population?
    nengo.Connection(model.op_state.output, model.count_res.mem1.gate,
                     transform=np.array([vocab.parse("CNT2").v * 1.5]))
    nengo.Connection(model.op_state.output, model.count_res.mem2.gate,
                     transform=np.array([vocab.parse("CNT1+NONE").v]))

    nengo.Connection(model.op_state.output, model.count_tot.mem1.gate,
                     transform=np.array([vocab.parse("CNT2").v * 1.5]))
    nengo.Connection(model.op_state.output, model.count_tot.mem2.gate,
                     transform=np.array([vocab.parse("CNT1+NONE").v]))


    # TODO: add the dot transform
    cortical_actions = spa.Actions(
        "count_res = q1*ONE",
        "count_fin = q2"
    )

    model.cortical = spa.Cortical(cortical_actions)

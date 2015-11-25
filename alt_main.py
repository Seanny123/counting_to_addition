import nengo
from nengo import spa
import numpy as np
import ipdb

from collections import OrderedDict

from inc_network import IncNet

D = 64
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
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

#vocab.add("ALLNUM", vocab.parse(join_num))

#vocab.add("TRANS", vocab.parse("CNT1+CNT2"))

num_vocab = vocab.create_subset(num_ord_filt.keys()+["NONE"])
state_vocab = vocab.create_subset(["NONE", "CNT1", "CNT2"])

model = spa.SPA(vocabs=[vocab], label="Count Net", seed=0)

with model:
    model.q1 = spa.State(D, vocab=num_vocab)
    model.q2 = spa.State(D, vocab=num_vocab)
    model.answer = spa.State(D, vocab=num_vocab)
    # TODO: add connection from question to answer, maybe intermediate pop?
    # Also going to need a bit of a gate...

    model.op_state = spa.State(D, vocab=state_vocab, feedback=1, feedback_synapse=0.2)

    model.count_res = IncNet(D, num_vocab, label="result")
    model.count_tot = IncNet(D, num_vocab, label="total")

    model.count_fin = spa.State(D, vocab=num_vocab, feedback=1)

    model.comp_tot_fin = spa.Compare(D)

    step = 0.1
    def q1_func(t):
        if(t < step):
            return "TWO"
        else:
            return "0"

    def q2_func(t):
        if(t < step):
            return "TWO"
        else:
            return "0"

    def op_state_func(t):
        if(t < 0.05):
            return "NONE"
        else:
            return "0"

    model.input = spa.Input(q1=q1_func, q2=q2_func)

    # TODO: add the max count
    """
    def bigger_func(t, x):
        v1 = num_ord_filt(vo.text(x[:D]).split(".")[1][2:])
        v2 = num_ord_filt(vo.text(x[D:]).split(".")[1][2:])

        if v1 < v2:
            return 0
        else:
            return 1

    bigger_finder = nengo.Ensemble(400, D, function=bigger_func)
    """

    actions = spa.Actions(
        # If the input isn't blank, read it in
        on_input=
        "(0.5*dot(q1, %s) + 0.5*dot(q2, %s) - 1.5*dot(op_state, CNT1+CNT2) + dot(op_state, NONE))/1.5 "
        "--> count_res = q1*ONE, count_tot = ONE, count_fin = q2, op_state = CNT2" % (join_num, join_num,),
        # load value into mem1
        load_1=
        "dot(op_state, CNT1) - comp_tot_fin "
        "--> op_state = CNT2",
        increment_2=
        "dot(op_state, CNT2) - comp_tot_fin "
        "--> op_state = CNT1,"
        "q1 = 0, q2 = 0",
        # If we're done incrementing write it to the answer
        answer=
        "1.5*comp_tot_fin - dot(op_state, CNT1+CNT2)"
        "--> answer = count_res, op_state = NONE"
    )

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    # So... Apparently, the gates aren't supposed to go to -1...
    # Maybe I should add a thresholding population?
    nengo.Connection(model.op_state.output, model.count_res.mem1.gate,
                     transform=np.array([vocab.parse("CNT2").v * 2]))
    nengo.Connection(model.op_state.output, model.count_res.mem2.gate,
                     transform=np.array([vocab.parse("CNT1+NONE").v * 2]))

    nengo.Connection(model.op_state.output, model.count_tot.mem1.gate,
                     transform=np.array([vocab.parse("CNT2").v * 2]))
    nengo.Connection(model.op_state.output, model.count_tot.mem2.gate,
                     transform=np.array([vocab.parse("CNT1+NONE").v * 2]))


    # TODO: add the dot transform
    cortical_actions = spa.Actions(
        "comp_tot_fin_A = count_fin",
        "comp_tot_fin_B = count_tot"
    )

    model.cortical = spa.Cortical(cortical_actions)

# I can't tell if it's the subvocab not working or my damn plots

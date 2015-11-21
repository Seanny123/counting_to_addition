import nengo
from nengo import spa

import numpy
from collections import OrderedDict

print(numpy.version.version)

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

vocab.add("ALLNUM", vocab.parse(join_num))

vocab.add("TRANS", vocab.parse("CNT1+CNT2"))

model = spa.SPA(vocabs=[vocab])

with model:
    model.q1 = spa.State(D)
    model.q2 = spa.State(D)
    model.answer = spa.State(D)
    # TODO: add connection from question to answer, maybe intermediate pop?
    # Also going to need a bit of a gate...

    model.op_state = spa.State(D)

    model.count_res = spa.State(D, feedback=1)
    model.res_mem = spa.State(D, feedback=1)
    model.count_tot = spa.State(D, feedback=1)
    model.tot_mem = spa.State(D, feedback=1)
    model.count_fin = spa.State(D, feedback=1)

    model.comp_tot_fin = spa.Compare(D)

    def q1_func(t):
        if(t < 0.1):
            return "TWO"
        else:
            return "0"

    def q2_func(t):
        if(t < 0.1):
            return "TWO"
        else:
            return "0"

    def op_state_func(t):
        if(t < 0.1):
            return "NONE"
        else:
            return "0"

    model.input = spa.Input(q1=q1_func, q2=q2_func, op_state=op_state_func)

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

    # reload
    actions = spa.Actions(
        # If the input isn't blank, read it in
        on_input=
        "0.5*dot(q1, ALLNUM) + 0.5*dot(q2, ALLNUM) - 1.5*dot(op_state, TRANS) + dot(op_state, NONE) "
        "--> count_res = q1*ONE, count_tot = ONE, count_fin = q2, op_state = CNT2",
        # If we have finished incrementing, keep incrementing
        increment_1=
        "2*dot(op_state, CNT1) - comp_tot_fin "
        "--> count_res = res_mem * ONE * 2, count_tot = tot_mem * ONE * 2, op_state = CNT2",
        increment_2=
        "2*dot(op_state, CNT2) - comp_tot_fin "
        "--> res_mem = count_res, tot_mem = count_tot, op_state = CNT1, "
        "q1 = NONE, q2 = NONE",
        # If we're done incrementing write it to the answer
        answer=
        "1.5*comp_tot_fin - dot(op_state, TRANS)"
        "--> answer = count_res, op_state = NONE"
    )

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    cortical_actions = spa.Actions(
        "comp_tot_fin_A = count_fin",
        "comp_tot_fin_B = count_tot"
    )

    model.cortical = spa.Cortical(cortical_actions)
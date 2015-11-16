import nengo
from nengo import spa
from collections import OrderedDict

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

model = spa.SPA(vocab)

with model:
    model.q1 = spa.State(D)
    model.q2 = spa.State(D)
    model.answer = spa.State(D)
    # TODO: add connection from question to answer, maybe intermediate pop?
    # Also going to need a bit of a gate...

    model.op_state = spa.State(D)

    model.count_res = spa.State(D, feedback=1)
    model.count_fin = spa.State(D, feedback=1)
    model.count_tot = spa.State(D, feedback=1)

    model.comp_tot_fin = spa.Compare(D)

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

    # Probably can't do this...
    actions = spa.Actions(
        # If the input isn't blank, read it in
        on_input="dot(q1, %s) + dot(q2, %s) - dot(op_state, COUNTING)--> count_res = q1*ONE, count_tot = ONE, count_fin = q2" % (join_num, join_num,),
        # If we have finished incrementing, keep incrementing
        increment="dot(op_state, COUNTING) - comp_tot_fin --> count_res = count_res * ONE, count_tot = count_tot * ONE",
        # If we're done incrementing write it to the answer
        answer="comp_tot_fin - dot(op_state, COUNTING) --> answer = count_res, op_state = NONE"
    )

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    cortical_actions = spa.Actions(
        "comp_tot_fin_A = count_fin",
        "comp_tot_fin_B = count_tot"
    )

    model.cortical = spa.Cortical(cortical_actions)
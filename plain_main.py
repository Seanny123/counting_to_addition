import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict

D = 64
less_D = 32
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
number_dict = {"ZERO":0, "ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 4
vocab.parse("ZERO")
number_list = number_ordered.keys()
for i in range(1, number_range):
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

join_num = "+".join(number_list[0:number_range])
num_ord_filt = OrderedDict(number_ordered.items()[:number_range])

print(join_num)

state_vocab = spa.Vocabulary(less_D)
state_vocab.parse("LOAD+INC+CMP+NONE")

num_vocab = vocab.create_subset(num_ord_filt.keys())

model = spa.SPA(vocabs=[vocab, state_vocab], label="Count Net", seed=0)

with model:
    model.q1 = spa.State(D, vocab=num_vocab)
    model.q2 = spa.State(D, vocab=num_vocab)
    model.answer = spa.State(D, vocab=num_vocab)
    # TODO: add connection from question to answer, maybe intermediate pop?
    # Also going to need a bit of a gate...

    model.op_state = spa.State(less_D, vocab=state_vocab, feedback=1)

    # TODO: Make this adaptively large
    input_keys = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR']
    output_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']

    model.res_assoc = spa.AssociativeMemory(input_vocab=num_vocab, output_vocab=num_vocab,
                                            input_keys=input_keys, output_keys=output_keys,
                                            wta_output=True)
    model.count_res = spa.State(D, vocab=num_vocab, feedback=1)
    model.res_mem = spa.State(D, vocab=num_vocab, feedback=1)

    model.tot_assoc = spa.AssociativeMemory(input_vocab=num_vocab, output_vocab=num_vocab,
                                            input_keys=input_keys, output_keys=output_keys,
                                            wta_output=True)
    model.count_tot = spa.State(D, vocab=num_vocab, feedback=1)
    model.tot_mem = spa.State(D, vocab=num_vocab, feedback=1)

    model.count_fin = spa.State(D, vocab=num_vocab, feedback=1)

    model.comp_tot_fin = spa.Compare(D)

    def q1_func(t):
        if(t < 0.05):
            return "ONE"
        else:
            return "0"

    def q2_func(t):
        if(t < 0.05):
            return "THREE"
        else:
            return "0"

    def op_state_func(t):
        if(t < 0.05):
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


    # Even with preset states this routing is fucked
    actions = spa.Actions(
        # If the input isn't blank, read it in
        on_input=
        "(0.5*dot(q1, %s) + 0.5*dot(q2, %s) + dot(op_state, NONE))/1.5 "
        "--> count_res = q1*ONE, count_tot = ONE, count_fin = q2, op_state = CMP" % (join_num, join_num,),
        # If we have finished incrementing, keep incrementing
        increment=
        "1.25*dot(op_state, INC) - comp_tot_fin - dot(op_state, LOAD)"
        "--> res_assoc = res_mem, tot_assoc = tot_mem, op_state = 0.75*CMP - 0.5*LOAD",
        load_mem=
        "1.25*dot(op_state, LOAD) - comp_tot_fin - dot(op_state, CMP)"
        "--> res_mem = count_res, tot_mem = count_tot, op_state = 0.75*INC - 0.5*CMP",
        # If not done, keep incrementing
        cmp_fail=
        "0.75*dot(op_state, CMP) - 0.5*comp_tot_fin "
        "--> op_state = 0.5*LOAD - NONE",
        # If we're done incrementing write it to the answer
        cmp_good=
        "0.25*dot(op_state, CMP) + comp_tot_fin"
        "--> answer = count_res, op_state=NONE"
    )

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    cortical_actions = spa.Actions(
        "count_res = res_assoc",
        "count_tot = tot_assoc",
        "comp_tot_fin_A = count_fin",
        "comp_tot_fin_B = count_tot"
    )

    model.cortical = spa.Cortical(cortical_actions)
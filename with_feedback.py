import nengo
from nengo import spa
from mem_net import MemNet

import numpy as np
from collections import OrderedDict

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

state_vocab = spa.Vocabulary(less_D)
state_vocab.parse("RUN+NONE")

num_vocab = vocab.create_subset(num_ord_filt.keys())

with spa.SPA(vocabs=[vocab, state_vocab], label="Count Net", seed=0) as model:
    model.q1 = spa.State(D, vocab=num_vocab)
    model.q2 = spa.State(D, vocab=num_vocab)
    model.answer = spa.State(D, vocab=num_vocab)
    # TODO: add connection from question to answer, maybe intermediate pop?
    # Also going to need a bit of a gate...

    # maybe this should be a mem block?
    model.op_state = MemNet(less_D, state_vocab, label="op_state")

    # TODO: Make this adaptively large
    input_keys = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR']
    output_keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']

    model.res_assoc = spa.AssociativeMemory(input_vocab=num_vocab, output_vocab=num_vocab,
                                            input_keys=input_keys, output_keys=output_keys,
                                            wta_output=True, input_scale=2.5)
    model.count_res = MemNet(D, num_vocab, label="count_res")
    model.res_mem = MemNet(D, num_vocab, label="res_mem")
    model.rmem_assoc = spa.AssociativeMemory(input_vocab=num_vocab,
                                             wta_output=True, input_scale=2.5)

    model.tot_assoc = spa.AssociativeMemory(input_vocab=num_vocab, output_vocab=num_vocab,
                                            input_keys=input_keys, output_keys=output_keys,
                                            wta_output=True, input_scale=2.5)
    model.count_tot = MemNet(D, num_vocab, label="count_tot")
    model.tot_mem = MemNet(D, num_vocab, label="tot_mem")
    model.tmem_assoc = spa.AssociativeMemory(input_vocab=num_vocab,
                                             wta_output=True, input_scale=2.5)

    model.count_fin = MemNet(D, num_vocab, label="count_fin")

    model.comp_tot_fin = spa.Compare(D)
    model.fin_assoc = spa.AssociativeMemory(input_vocab=num_vocab,
                                             wta_output=True)

    model.comp_load_res = spa.Compare(D)
    model.comp_inc_res = spa.Compare(D)
    model.comp_assoc = spa.AssociativeMemory(input_vocab=num_vocab,
                                             wta_output=True, input_scale=2.5)

    def q1_func(t):
        if(t < 0.07):
            return "ONE"
        elif(0.2 < t < 0.3):
            return "TWO"
        else:
            return "0"

    def q2_func(t):
        if(t < 0.07):
            return "THREE"
        elif(0.2 < t < 0.3):
            return "TWO"
        else:
            return "0"

    def op_state_func(t):
        if(t < 0.05):
            return "NONE"
        elif(0.2 < t < 0.25):
            return "NONE - RUN"
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

    main_actions = spa.Actions(
        ## If the input isn't blank, read it in
        on_input=
        "(dot(q1, %s) + dot(q2, %s))/2 "
        "--> count_res = q1*ONE, count_tot = ONE, fin_assoc = q2, op_state = RUN" % (join_num, join_num,),
        ## If not done, prepare next increment
        cmp_fail=
        "1.25*dot(op_state, RUN) - 0.5*comp_tot_fin + comp_inc_res - comp_load_res"
        "--> op_state = 0.5*RUN - NONE, rmem_assoc = count_res, tmem_assoc = count_tot, "
        "count_res_gate = CLOSE, count_tot_gate = CLOSE, op_state_gate = CLOSE, count_fin_gate = CLOSE, "
        "comp_load_res_A = res_mem, comp_load_res_B = comp_assoc, comp_assoc = count_res",
        ## If we're done incrementing write it to the answer
        cmp_good=
        "0.5*dot(op_state, RUN) + 0.5*comp_tot_fin"
        "--> answer = count_res, op_state=NONE",
        ## Increment memory transfer
        increment=
        "0.3*dot(op_state, RUN) + 1.2*comp_load_res - comp_inc_res"
        "--> res_assoc = res_mem, tot_assoc = tot_mem, "
        "res_mem_gate = CLOSE, tot_mem_gate = CLOSE, op_state_gate = CLOSE, count_fin_gate = CLOSE, "
        "comp_load_res_A = 0.75*ONE, comp_load_res_B = 0.75*ONE, "
        "comp_inc_res_A = comp_assoc, comp_assoc = res_mem*ONE, comp_inc_res_B = count_res",
    )

    model.bg_main = spa.BasalGanglia(main_actions)
    model.thal_main = spa.Thalamus(model.bg_main)

    # had to put the assoc connections here because bugs
    # ideally they should be routable
    cortical_actions = spa.Actions(
        "res_mem = rmem_assoc, tot_mem = tmem_assoc",
        "count_res = res_assoc, count_tot = tot_assoc",
        "count_fin = fin_assoc",
        "comp_tot_fin_A = count_fin",
        "comp_tot_fin_B = count_tot",
    )

    model.cortical = spa.Cortical(cortical_actions)

    model.speech = spa.State(D)
    model.fast_ans = spa.State(D)

    feedback_actions = spa.Actions(
        fast="speech = fast_ans",
        slow="speech = answer"
    )
    model.feedback_bg = spa.BasalGanglia(feedback_actions)
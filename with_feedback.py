import nengo
from nengo import spa
from nengo.dists import Exponential, Choice, Uniform
from mem_net import MemNet
from adder_env import create_adder_env
from constants import *

import numpy as np
from collections import OrderedDict
import itertools
import ipdb

## Generate the vocab awkwardly
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_list = number_ordered.keys()
for i in range(number_range):
    print(number_list[i])
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

join_num = "+".join(number_list[0:number_range])

# TODO: Filter for max count
q_list = []
ans_list = []
for val in itertools.product(number_list, number_list):
    if val[0] >= val[1]:
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

# TESTING
q_list[0] = q_list[4]
ans_list[0] = ans_list[4]

## Generate specialised vocabs
state_vocab = spa.Vocabulary(less_D)
state_vocab.parse("RUN+NONE")

with nengo.Network(label="Root Net", seed=0) as model:
    env = create_adder_env(q_list, ans_list, state_vocab.parse("NONE").v, vocab)

    with spa.SPA(vocabs=[vocab, state_vocab], label="Count Net", seed=0) as slow_net:
        slow_net.q1 = spa.State(D, vocab=vocab)
        slow_net.q2 = spa.State(D, vocab=vocab)

        slow_net.answer = spa.State(D, vocab=vocab)

        slow_net.op_state = MemNet(less_D, state_vocab, label="op_state")

        # TODO: Make this adaptively large
        input_keys = number_list[:-1]
        output_keys = number_list[1:]

        slow_net.res_assoc = spa.AssociativeMemory(input_vocab=vocab, output_vocab=vocab,
                                                input_keys=input_keys, output_keys=output_keys,
                                                wta_output=True)
        slow_net.count_res = MemNet(D, vocab, label="count_res")
        slow_net.res_mem = MemNet(D, vocab, label="res_mem")
        slow_net.rmem_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                    wta_output=True)

        slow_net.tot_assoc = spa.AssociativeMemory(input_vocab=vocab, output_vocab=vocab,
                                                input_keys=input_keys, output_keys=output_keys,
                                                wta_output=True)
        slow_net.count_tot = MemNet(D, vocab, label="count_tot")
        slow_net.tot_mem = MemNet(D, vocab, label="tot_mem")
        slow_net.tmem_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                    wta_output=True)

        slow_net.ans_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                 wta_output=True)

        slow_net.count_fin = MemNet(D, vocab, label="count_fin")

        slow_net.tot_fin_simi = spa.State(1)
        slow_net.comp_tot_fin = spa.Compare(D)
        slow_net.fin_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                   wta_output=True)

        slow_net.comp_load_res = spa.Compare(D)
        slow_net.comp_inc_res = spa.Compare(D)
        slow_net.comp_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                    wta_output=True)

        main_actions = spa.Actions(
            ## If the input isn't blank, read it in
            on_input=
            "(dot(q1, %s) + dot(q2, %s))/2 "
            "--> count_res = q1*ONE, count_tot = ONE, fin_assoc = 2.5*q2, op_state = RUN" % (join_num, join_num,),
            ## If not done, prepare next increment
            cmp_fail=
            "dot(op_state, RUN) - tot_fin_simi + 1.25*comp_inc_res - comp_load_res"
            "--> op_state = 0.5*RUN - NONE, rmem_assoc = 2.5*count_res, tmem_assoc = 2.5*count_tot, "
            "count_res_gate = CLOSE, count_tot_gate = CLOSE, op_state_gate = CLOSE, count_fin_gate = CLOSE, "
            "comp_load_res_A = res_mem, comp_load_res_B = comp_assoc, comp_assoc = 2.5*count_res",
            ## If we're done incrementing write it to the answer
            cmp_good=
            "0.5*dot(op_state, RUN) + tot_fin_simi"
            "--> ans_assoc = 8*count_res, op_state = 0.5*RUN,"
            "count_res_gate = CLOSE, count_tot_gate = CLOSE, op_state_gate = CLOSE, count_fin_gate = CLOSE",
            ## Increment memory transfer
            increment=
            "0.3*dot(op_state, RUN) + 1.2*comp_load_res - comp_inc_res"
            "--> res_assoc = 2.5*res_mem, tot_assoc = 2.5*tot_mem, "
            "res_mem_gate = CLOSE, tot_mem_gate = CLOSE, op_state_gate = CLOSE, count_fin_gate = CLOSE, "
            "comp_load_res_A = 0.75*ONE, comp_load_res_B = 0.75*ONE, "
            "comp_inc_res_A = comp_assoc, comp_assoc = 2.5*res_mem*ONE, comp_inc_res_B = count_res",
        )

        slow_net.bg_main = spa.BasalGanglia(main_actions)
        slow_net.thal_main = spa.Thalamus(slow_net.bg_main)

        thr = 0.25
        thresh_ens = nengo.Ensemble(100, 1, encoders=Choice([[1]]), intercepts=Exponential(scale=(1 - thr) / 5.0, shift=thr, high=1),
            eval_points=Uniform(thr, 1.1), n_eval_points=5000)
        nengo.Connection(slow_net.comp_tot_fin.output, thresh_ens)
        nengo.Connection(thresh_ens, slow_net.tot_fin_simi.input)

        ans_boost = nengo.networks.Product(200, dimensions=D, input_magnitude=2)
        nengo.Connection(slow_net.ans_assoc.output, ans_boost.A)
        nengo.Connection(thresh_ens, ans_boost.B,
                         transform=np.ones((D,1)))
        nengo.Connection(ans_boost.output, slow_net.answer.input, transform=2.5)


        # had to put the assoc connections here because bugs
        # ideally they should be routable
        cortical_actions = spa.Actions(
            "res_mem = rmem_assoc, tot_mem = tmem_assoc",
            "count_res = res_assoc, count_tot = tot_assoc",
            "count_fin = fin_assoc",
            "comp_tot_fin_A = count_fin",
            "comp_tot_fin_B = 0.5*count_tot",
        )

        slow_net.cortical = spa.Cortical(cortical_actions)

    nengo.Connection(env.q_in[D:], slow_net.q1.input)
    nengo.Connection(env.q_in[:D], slow_net.q2.input)
    nengo.Connection(env.op_in, slow_net.op_state.mem.input)

    with spa.SPA(vocabs=[vocab], label="Mem Net", seed=0) as fast_net:
        fast_net.final_cleanup = spa.AssociativeMemory(input_vocab=vocab,
                                                    threshold=0.2,
                                                    wta_output=True)

        adder = nengo.networks.AssociativeMemory(input_vectors=q_list, n_neurons=n_neurons)
        adder.add_wta_network()
        
        adder_out = nengo.Ensemble(n_neurons*8, len(ans_list))
        nengo.Connection(adder.elem_output, adder_out)
        conf_in = nengo.Ensemble(n_neurons*8, len(q_list))
        nengo.Connection(adder.elem_output, conf_in)

        fast_net.recall = spa.State(D)
        fast_net.conf = spa.State(1)

        nengo.Connection(env.env_keys, adder.input)

        conn_out = nengo.Connection(adder_out, fast_net.recall.input, learning_rule_type=nengo.PES(1e-5),
                                    function=lambda x: np.zeros(D))

        # Create the error population and node
        # These will come from the environment
        error = nengo.Ensemble(n_neurons*8, D)
        nengo.Connection(env.learning, error.neurons, transform=[[10.0]]*n_neurons*8,
                         synapse=None)

        def err_func(t, x):
            mag = np.linalg.norm(x[D])
            if mag < 0.1:
                return -1*x[-1]
            else:
                return 1*x[-1]

        err_mag = nengo.Node(err_func, size_in=D+1)
        nengo.Connection(error, err_mag[:D], synapse=0.01)
        nengo.Connection(env.learning, err_mag[-1])

        # Calculate the error and use it to drive the PES rule
        nengo.Connection(env.get_ans, error, transform=-1, synapse=None)
        nengo.Connection(fast_net.recall.output, error, synapse=None)
        nengo.Connection(error, conn_out.learning_rule)

        # Let confidence be inversely proportional to the magnitude of the error
        conn_conf = nengo.Connection(conf_in, fast_net.conf.input, learning_rule_type=nengo.PES(1e-5),
                                     function=lambda x: 0.2)
        nengo.Connection(err_mag, conn_conf.learning_rule)

        # TODO: This probably needs a cleanup...
        fast_net.speech = MemNet(D, vocab, label="speech")

        feedback_actions = spa.Actions(
            fast="conf --> speech = recall",
            slow="1 - conf --> speech = 2.5*final_cleanup"
        )
        fast_net.feedback_bg = spa.BasalGanglia(feedback_actions)
        fast_net.feedback_thal = spa.Thalamus(fast_net.feedback_bg)

    nengo.Connection(slow_net.answer.output, fast_net.final_cleanup.input)
    nengo.Connection(fast_net.speech.mem.output, env.set_ans)
    # I don't know if sustaining this is absolutely necessary...
    # The actual answer is comming out of the env anyways
    nengo.Connection(env.reset, fast_net.speech.mem.reset, synapse=None)
    # reset all the counting network
    nengo.Connection(env.count_reset, slow_net.count_res.mem.reset, synapse=None)
    nengo.Connection(env.count_reset, slow_net.count_fin.mem.reset, synapse=None)
    nengo.Connection(env.count_reset, slow_net.count_tot.mem.reset, synapse=None)
    nengo.Connection(env.count_reset, slow_net.op_state.mem.reset, synapse=None)
    nengo.Connection(env.gate, fast_net.speech.mem.gate, synapse=None)

    get_data = "file"
    if get_data == "probe":
        p_keys = nengo.Probe(env.env_keys, synapse=None)
        p_learning = nengo.Probe(env.learning, synapse=None)
        p_error = nengo.Probe(error, synapse=0.005)
        p_conf = nengo.Probe(fast_net.conf.output)
        p_recall = nengo.Probe(fast_net.recall.output, synapse=None)
        p_final_ans = nengo.Probe(fast_net.final_cleanup.output)
        p_speech = nengo.Probe(fast_net.speech.mem.output)

        p_bg_in = nengo.Probe(slow_net.bg_main.input)
        p_bg_out = nengo.Probe(slow_net.bg_main.output)

        p_count_res = nengo.Probe(slow_net.count_res.mem.output)
        p_count_fin = nengo.Probe(slow_net.count_fin.mem.output)
        p_count_tot = nengo.Probe(slow_net.count_tot.mem.output)
        p_ans_assoc = nengo.Probe(slow_net.ans_assoc.output)
        p_thres_ens = nengo.Probe(thresh_ens)
    else:
        def file_func(filename):
            fi = open("data/%s" %filename, "w")
            def f(t, x):
                fi.write("%s\n" %x)
            return f

        p_keys = nengo.Node(file_func("p_keys"), size_in=2*D)
        nengo.Connection(env.env_keys, p_keys, synapse=None)
        p_learning = nengo.Node(file_func("p_learning"), size_in=1)
        nengo.Connection(env.learning, p_learning, synapse=None)
        p_error = nengo.Node(file_func("p_error"), size_in=D)
        nengo.Connection(error, p_error, synapse=0.005)
        p_conf = nengo.Node(file_func("p_conf"), size_in=1)
        nengo.Connection(fast_net.conf.output, p_conf)
        p_recall = nengo.Node(file_func("p_recall"), size_in=D)
        nengo.Connection(fast_net.recall.output, p_recall, synapse=None)
        p_final_ans = nengo.Node(file_func("p_final_ans"), size_in=D)
        nengo.Connection(fast_net.final_cleanup.output, p_final_ans)
        p_speech = nengo.Node(file_func("p_speech"), size_in=D)
        nengo.Connection(fast_net.speech.mem.output, p_speech)

        p_bg_in = nengo.Node(file_func("p_bg_in"), size_in=4)
        nengo.Connection(slow_net.bg_main.input, p_bg_in)
        p_bg_out = nengo.Node(file_func("p_bg_out"), size_in=4)
        nengo.Connection(slow_net.bg_main.output, p_bg_out)

        p_count_res = nengo.Node(file_func("p_count_res"), size_in=D)
        nengo.Connection(slow_net.count_res.mem.output, p_count_res)
        p_count_fin = nengo.Node(file_func("p_count_fin"), size_in=D)
        nengo.Connection(slow_net.count_fin.mem.output, p_count_fin)
        p_count_tot = nengo.Node(file_func("p_count_tot"), size_in=D)
        nengo.Connection(slow_net.count_tot.mem.output, p_count_tot)
        p_ans_assoc = nengo.Node(file_func("p_ans_assoc"), size_in=D)
        nengo.Connection(slow_net.ans_assoc.output, p_ans_assoc)
        p_thres_ens = nengo.Node(file_func("p_thres_ens"), size_in=1)
        nengo.Connection(thresh_ens, p_thres_ens)

print("Building")
sim = nengo.Simulator(model, dt=dt)

print("Running")
while env.env_cls.questions_answered < 2000:
    sim.step()
    if env.env_cls.time_since_last_answer > 7.0:
        print("UH OH")
        ipdb.set_trace()

ipdb.set_trace()

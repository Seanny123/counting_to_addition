# final full network version

from mem_net import MemNet
from adder_env import create_adder_env
from constants import *
from hetero_mem import build_hetero_mem, encoders
from utils import gen_vocab, gen_env_list

import nengo
from nengo import spa
from nengo.presets import ThresholdingEnsembles
import numpy as np
import ipdb

thresh_conf = ThresholdingEnsembles(0.25)

## Learning rates
pes_rate = 0.001
voja_rate = 0.005

## Generate the vocab
rng = np.random.RandomState(SEED)
number_dict = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
               "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9}

# This should be set to 9 for the actual final test
max_sum = 9
max_num = max_sum - 2

number_list, _ = gen_vocab(number_dict, max_num, D, rng)

# load a vocabulary that previously has few errors
# Ideally, this network would work with every vocabulary, but sometimes the vectors generated are too close together
# in that case, the manually tuned basal ganglia weights won't work anymore.
# Future work will include learning the basal ganglia weights over time so they work with every possible vocabulary
vo_load = np.load("data/good_slow_run/vocabslow.npz")
vocab = spa.Vocabulary(10)
for key, val in zip(number_list, vo_load['vocab']):
    vocab.add(key, val)

join_num = "+".join(number_list[0:max_num])

## Create inputs and expected outputs
q_list, q_norm_list, ans_list = gen_env_list(number_dict, number_list, vocab, max_sum)

## Generate specialised vocabs
state_vocab = spa.Vocabulary(less_D, rng=rng)
state_vocab.parse("RUN+NONE")

with nengo.Network(label="Root Net", seed=SEED) as model:
    env = create_adder_env(q_list, q_norm_list, ans_list, state_vocab.parse("NONE").v, vocab)

    with spa.SPA(vocabs=[vocab, state_vocab], label="Count Net", seed=SEED) as slow_net:
        slow_net.q1 = spa.State(D, vocab=vocab)
        slow_net.q2 = spa.State(D, vocab=vocab)

        slow_net.answer = spa.State(D, vocab=vocab)

        slow_net.op_state = MemNet(less_D, state_vocab, label="op_state")

        input_keys = number_list[:-1]
        output_keys = number_list[1:]

        ### Result circuit
        ## Incrementing memory
        slow_net.res_assoc = spa.AssociativeMemory(input_vocab=vocab, output_vocab=vocab,
                                                   input_keys=input_keys, output_keys=output_keys,
                                                   wta_output=True)
        ## Starting memory
        slow_net.count_res = MemNet(D, vocab, label="count_res")
        ## Increment result memory
        slow_net.res_mem = MemNet(D, vocab, label="res_mem")
        ## Cleanup memory
        slow_net.rmem_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                    wta_output=True)

        ### Total circuit
        ## Total memory
        slow_net.tot_assoc = spa.AssociativeMemory(input_vocab=vocab, output_vocab=vocab,
                                                   input_keys=input_keys, output_keys=output_keys,
                                                   wta_output=True)
        ## Starting memory
        slow_net.count_tot = MemNet(D, vocab, label="count_tot")
        ## Increment result memory
        slow_net.tot_mem = MemNet(D, vocab, label="tot_mem")
        ## Cleanup memory
        slow_net.tmem_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                    wta_output=True)

        slow_net.ans_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                   wta_output=True)

        ## The memory that says when to stop incrementing
        slow_net.count_fin = MemNet(D, vocab, label="count_fin")

        ### Comparison circuit
        ## State for easier insertion into Actions after threshold
        slow_net.tot_fin_simi = spa.State(1)
        slow_net.comp_tot_fin = spa.Compare(D)

        # this network is only used during the on_input action, is it really necessary?
        slow_net.fin_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                   wta_output=True)

        ### Compares that set the speed of the increment
        ## Compare for loading into start memory
        slow_net.comp_load_res = spa.Compare(D)
        ## Compare for loading into incrementing memory
        slow_net.comp_inc_res = spa.Compare(D)
        ## Cleanup for compare
        slow_net.comp_assoc = spa.AssociativeMemory(input_vocab=vocab,
                                                    wta_output=True)

        ## Increment for compare and input
        slow_net.gen_inc_assoc = spa.AssociativeMemory(input_vocab=vocab, output_vocab=vocab,
                                                       input_keys=input_keys, output_keys=output_keys,
                                                       wta_output=True)

        main_actions = spa.Actions(
            ## If the input isn't blank, read it in
            on_input=
            "(dot(q1, %s) + dot(q2, %s))/2 "
            "--> count_res = gen_inc_assoc, gen_inc_assoc = q1, count_tot = ONE, count_fin = fin_assoc, fin_assoc = 2.5*q2, op_state = RUN" % (
            join_num, join_num,),
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
            "comp_inc_res_A = gen_inc_assoc, gen_inc_assoc = 2.5*res_mem, comp_inc_res_B = count_res",
        )

        slow_net.bg_main = spa.BasalGanglia(main_actions)
        slow_net.thal_main = spa.Thalamus(slow_net.bg_main)

        ## Threshold preventing premature influence
        with thresh_conf:
            thresh_ens = nengo.Ensemble(100, 1)
        # prevents from comp_tot_fin similarity into the action selection
        nengo.Connection(slow_net.comp_tot_fin.output, thresh_ens)
        nengo.Connection(thresh_ens, slow_net.tot_fin_simi.input)

        # Because the answer is being continuously output, we've got to threshold it by comp_tot_fin
        ans_boost = nengo.networks.Product(200, dimensions=D, input_magnitude=2)
        ans_boost.label = "ans_boost"
        nengo.Connection(slow_net.ans_assoc.output, ans_boost.A)
        nengo.Connection(thresh_ens, ans_boost.B,
                         transform=np.ones((D, 1)))
        nengo.Connection(ans_boost.output, slow_net.answer.input, transform=2.5)

        # had to put the assoc connections here because bugs
        # ideally they should be routable
        cortical_actions = spa.Actions(
            "res_mem = rmem_assoc, tot_mem = tmem_assoc",
            "count_res = res_assoc, count_tot = tot_assoc",
            "comp_tot_fin_A = count_fin",
            "comp_tot_fin_B = 0.5*count_tot",
        )

        slow_net.cortical = spa.Cortical(cortical_actions)

    nengo.Connection(env.q_in[D:], slow_net.q1.input)
    nengo.Connection(env.q_in[:D], slow_net.q2.input)
    nengo.Connection(env.op_in, slow_net.op_state.mem.input)

    with spa.SPA(vocabs=[vocab], label="Fast Net", seed=SEED) as fast_net:

        ## Generate hetero mem
        K = 400
        # This is usually calculated
        c = 0.51
        e = encoders(np.array(q_norm_list), K, rng)
        fast_net.het_mem = build_hetero_mem(D * 2, D, e, c, pes_rate=pes_rate, voja_rate=voja_rate)

        ## Calculate the error from the environment and use it to drive the decoder learning
        # Create the error population
        error = nengo.Ensemble(n_neurons * 8, D)
        nengo.Connection(env.learning, error.neurons, transform=[[10.0]] * n_neurons * 8,
                         synapse=None)
        nengo.Connection(env.get_ans, error, transform=-1, synapse=None)
        nengo.Connection(fast_net.het_mem.output, error, synapse=None)
        nengo.Connection(error, fast_net.het_mem.out_conn.learning_rule)

        # encoder learning should only happen while decoder learning is happening
        nengo.Connection(env.learning, fast_net.het_mem.in_conn.learning_rule,
                         synapse=None)


        ## Calculate the similarity of the input and let it drive the confidence
        def get_mag(t, x):
            return np.max(np.dot(vocab.vectors, x))


        fast_net.conf = spa.State(1)
        # TODO: This should really be an ensemble...
        mag = nengo.Node(get_mag, size_in=D, size_out=1)
        nengo.Connection(fast_net.het_mem.output, mag)
        # It should be proportional to a match to one of the given vocabs
        conn_conf = nengo.Connection(mag, fast_net.conf.input)

        ## Final answer components
        # Final answer output
        # fast_net.speech = MemNet(D, vocab, label="speech")
        fast_net.speech = spa.State(D)
        # The final cleanup before outputting the answer
        fast_net.final_cleanup = spa.AssociativeMemory(input_vocab=vocab,
                                                       threshold=0.2,
                                                       wta_output=True)

        ## connect the output of the memory to a state for easier manipulation
        fast_net.recall = spa.State(D)
        nengo.Connection(fast_net.het_mem.output, fast_net.recall.input)

        feedback_actions = spa.Actions(
            fast="conf --> speech = recall",
            slow="1 - conf --> speech = 2.5*final_cleanup"
        )
        fast_net.feedback_bg = spa.BasalGanglia(feedback_actions)
        fast_net.feedback_thal = spa.Thalamus(fast_net.feedback_bg)

    ## Final answer connections
    nengo.Connection(slow_net.answer.output, fast_net.final_cleanup.input)
    nengo.Connection(fast_net.speech.output, env.set_ans)

    nengo.Connection(env.env_norm_keys, fast_net.het_mem.input)

    ## reset all the counting network
    nengo.Connection(env.count_reset, slow_net.count_res.mem.reset, synapse=None)
    nengo.Connection(env.count_reset, slow_net.count_fin.mem.reset, synapse=None)
    nengo.Connection(env.count_reset, slow_net.count_tot.mem.reset, synapse=None)
    nengo.Connection(env.count_reset, slow_net.op_state.mem.reset, synapse=None)

    get_data = "probe"
    debug_probes = True
    if get_data == "probe":
        p_keys = nengo.Probe(env.env_keys, synapse=None, sample_every=0.025)
        p_error = nengo.Probe(error, synapse=0.005, sample_every=0.01)
        p_recall = nengo.Probe(fast_net.recall.output, synapse=0.005, sample_every=0.01)

        p_count_res = nengo.Probe(slow_net.count_res.mem.output, synapse=0.005, sample_every=0.005)
        p_count_fin = nengo.Probe(slow_net.count_fin.mem.output, synapse=0.005, sample_every=0.005)
        p_count_tot = nengo.Probe(slow_net.count_tot.mem.output, synapse=0.005, sample_every=0.005)

        if debug_probes:
            p_final_ans = nengo.Probe(fast_net.final_cleanup.output, synapse=0.005, sample_every=0.01)
            p_speech = nengo.Probe(fast_net.speech.output, synapse=0.005, sample_every=0.01)

            p_bg_in = nengo.Probe(slow_net.bg_main.input, sample_every=0.01)
            p_bg_out = nengo.Probe(slow_net.bg_main.output, sample_every=0.01)

            p_ans_assoc = nengo.Probe(slow_net.ans_assoc.output, synapse=0.005, sample_every=0.01)
            p_thres_ens = nengo.Probe(thresh_ens, sample_every=0.01)


print("Building")
sim = nengo.Simulator(model, dt=dt)

print("Running")
while env.env_cls.questions_answered < 1175:
    sim.step()
    if env.env_cls.time_since_last_answer > 7.0:
        print("UH OH")
        ipdb.set_trace()


np.savez_compressed("data/paperslowlong_count_data", p_count_res=sim.data[p_count_res], p_count_fin=sim.data[p_count_fin], p_count_tot=sim.data[p_count_tot])

np.savez_compressed("data/paperslowlong_learning_data", p_keys=sim.data[p_keys], p_recall=sim.data[p_recall], p_error=sim.data[p_error])

np.savez_compressed("data/vocabslowlong", vals=vocab.keys, vecs=vocab.vectors)

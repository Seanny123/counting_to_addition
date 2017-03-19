# final full network version of the "fast net"

import nengo
from nengo import spa
from nengo.dists import Exponential, Choice, Uniform
from mem_net import MemNet
from adder_env import create_adder_env
from constants import *
from hetero_mem import *

import numpy as np
from collections import OrderedDict
import itertools

## Generate the vocab
rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, rng=rng)
number_dict = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
               "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))
# This should be set to 10 for the actual final test
number_range = 4
number_list = number_ordered.keys()


def nearest(d):
    from scipy.linalg import sqrtm
    p = nengo.dists.UniformHypersphere(surface=True).sample(d, d)
    return np.dot(p, np.linalg.inv(sqrtm(np.dot(p.T, p))))
orth_vecs = nearest(D)

for i in range(number_range):
    print(number_list[i])
    vocab.add(number_list[i], orth_vecs[i])

join_num = "+".join(number_list[0:number_range])

## Create inputs and expected outputs
q_list = []
q_norm_list = []
ans_list = []
M = 0
for val in itertools.product(number_list, number_list):
    # Filter for min count # TODO: This might be backwards...
    if val[0] >= val[1]:
        ans_val = number_dict[val[0]] + number_dict[val[1]]
        if ans_val <= number_range:
            q_list.append(
                np.concatenate(
                    (vocab.parse(val[0]).v, vocab.parse(val[1]).v)
                )
            )
            q_norm_list.append(
                np.concatenate(
                    (vocab.parse(val[0]).v, vocab.parse(val[1]).v)
                ) / np.sqrt(2.0)
            )
            assert np.allclose(np.linalg.norm(q_norm_list[-1]), 1)
            ans_list.append(
                vocab.parse(number_list[ans_val-1]).v
            )
            M += 1
            print("%s+%s=%s" %(val[0], val[1], number_list[ans_val-1]))

# TESTING
q_list[0] = q_list[2]
ans_list[0] = ans_list[2]
q_norm_list[0] = q_norm_list[2]
q_list[1] = q_list[2]
ans_list[1] = ans_list[2]
q_norm_list[1] = q_norm_list[2]

## Generate specialised vocabs
state_vocab = spa.Vocabulary(less_D)
state_vocab.parse("RUN+NONE")

with nengo.Network(label="Root Net", seed=0) as model:
    env = create_adder_env(q_list, q_norm_list, ans_list, state_vocab.parse("NONE").v, vocab)

    with spa.SPA(vocabs=[vocab], label="Fast Net", seed=0) as fast_net:

        ## Generate hetero mem
        K = 400
        # This is usually calculated
        c = 0.51
        e = encoders(np.array(q_norm_list), K, rng)
        fast_net.het_mem = build_hetero_mem(D*2, D, e, c)

        ## Calculate the error from the environment and use it to drive the decoder learning
        # Create the error population
        error = nengo.Ensemble(n_neurons*8, D)
        nengo.Connection(env.learning, error.neurons, transform=[[10.0]]*n_neurons*8,
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
        #fast_net.speech = MemNet(D, vocab, label="speech")
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

    with spa.SPA(vocabs=[vocab], label="Slow Net", seed=0) as slow_net:
        slow_net.fake_answer = spa.State(D)
        slow_net.q1 = spa.State(D, vocab=vocab)
        slow_net.q2 = spa.State(D, vocab=vocab)

        def fake_func(t):
            if 0.6 > t > 0.5:
                return "FOUR"
            else:
                return '0'

        slow_net.fake_in = spa.Input(fake_answer=fake_func)

    nengo.Connection(env.q_in[D:], slow_net.q1.input)
    nengo.Connection(env.q_in[:D], slow_net.q2.input)

    ## Final answer connections
    nengo.Connection(slow_net.fake_answer.output, fast_net.final_cleanup.input)
    nengo.Connection(fast_net.speech.output, env.set_ans)

    nengo.Connection(env.env_norm_keys, fast_net.het_mem.input)

sim = nengo.Simulator(model, dt=dt)

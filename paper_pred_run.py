# no voja, single concatenated vector input
# used for making plot showing decoders continually switching back and forth

# make sure it messes up in the way you expect
# get the decoder data
# try to show the decoder values changing
# - look for decoders who's values change with every question
import itertools

from utils import gen_env_list, gen_vocab, add_to_env_list
from hetero_mem import build_hetero_mem, encoders
from constants import n_neurons, D, dt

import nengo
from nengo import spa
import numpy as np

import ipdb

plot_res = True

# Learning rates
pes_rate = 0.01
voja_rate = 0.003

## Generate the vocab
rng = np.random.RandomState(0)
number_dict = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
               "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9}

max_sum = 9
max_num = max_sum - 2

number_list, vocab = gen_vocab(number_dict, max_num, D, rng)

join_num = "+".join(number_list[0:max_num])

## Create inputs and expected outputs
q_list, q_norm_list, ans_list = gen_env_list(number_dict, number_list, vocab, max_sum)


def filt_addends(get, avoid, cond="or"):
    """Make list of numbers contain one type of addend, but not another
    Note: could probably determine this analytically"""
    q_l = []
    q_n_l = []
    ans_l = []

    for val in itertools.product(number_list, number_list):
        if cond == "or":
            get_val = val[0] in get or val[1] in get
        elif cond == "and":
            get_val = val[0] in get and val[1] in get
        else:
            raise NotImplementedError

        avoid_val = val[0] in avoid or val[1] in avoid

        ans_val = number_dict[val[0]] + number_dict[val[1]]
        size_lim = ans_val <= max_sum

        if get_val and size_lim and not avoid_val:
            add_to_env_list(val, ans_val, q_l, q_n_l, ans_l, number_list, vocab)
            print("%s+%s=%s" % (val[0], val[1], number_list[ans_val - 1]))

    return q_l, q_n_l, ans_l

num_items = len(q_list)
period = 0.3

# Intended result is that 3+5 is learned faster than 3+4
train_get = "FIVE"
train_avoid = "FOUR"
test_avoid = "THREE"

print("\n\nTraining questions")
_, train_qs_nrm, train_ans = filt_addends([train_get], [train_avoid, test_avoid])
print("Good tests")
_, gtest_qs_nrm, gtest_ans = filt_addends([train_get, test_avoid], [train_avoid], cond="and")
print("Fail tests")
_, ftest_qs_nrm, ftest_ans = filt_addends([train_avoid], [train_get, test_avoid])
ftest_qs_nrm[-1], ftest_qs_nrm[0] = ftest_qs_nrm[0], ftest_qs_nrm[-1]
ftest_ans[-1], ftest_ans[0] = ftest_ans[0], ftest_ans[-1]
run_time = (len(train_ans) + len(gtest_ans) + len(ftest_ans)) * period


def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period/dt))
    if i_every != period/dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def f(t):
        i = int(round((t - dt)/dt))  # t starts at dt
        return x[(i/i_every) % len(x)]

    return f


def create_cc_func(train_list, gtest_list, ftest_list, size=D):

    train_time = len(train_list) * period
    gtest_time = train_time + len(gtest_list) * period
    ftest_time = gtest_time + len(ftest_list) * period

    def cc(t):
        i = int(round((t - dt) / dt)) / int(round(period/dt))
        if t < train_time:
            return train_list[i]
        elif t < gtest_time:
            return gtest_list[i - len(train_list)]
        elif t < ftest_time:
            return ftest_list[i - len(train_list) - len(gtest_list)]
        else:
            return np.zeros(size)

    return cc


with spa.SPA(vocabs=[vocab], label="Fast Net", seed=0) as model:
    env_keys = nengo.Node(create_cc_func(train_qs_nrm, gtest_qs_nrm, ftest_qs_nrm, 2*D))
    env_values = nengo.Node(create_cc_func(train_ans, gtest_ans, ftest_ans, D))

    recall = nengo.Node(size_in=D)
    learning = nengo.Node([0])

    ## Generate hetero mem
    K = 400
    # This is usually calculated
    c = 0.51
    e = encoders(np.array(q_norm_list), K, rng)
    het_mem = build_hetero_mem(D*2, D, e, c, pes_rate=pes_rate, voja_rate=voja_rate)

    nengo.Connection(env_keys, het_mem.input, synapse=None)
    nengo.Connection(het_mem.output, recall)

    # Create the error population
    error = nengo.Ensemble(n_neurons*8, D)
    nengo.Connection(learning, error.neurons, transform=[[10.0]]*n_neurons*8,
                     synapse=None)
    nengo.Connection(learning, het_mem.in_conn.learning_rule, synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(env_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, het_mem.out_conn.learning_rule)

    # Setup probes
    p_keys = nengo.Probe(env_keys, synapse=None, sample_every=0.01)
    p_values = nengo.Probe(env_values, synapse=None, sample_every=0.01)
    p_learning = nengo.Probe(learning, synapse=None, sample_every=0.01)
    p_error = nengo.Probe(error, synapse=0.01)
    p_recall = nengo.Probe(recall, synapse=None, sample_every=0.01)
    p_weights = nengo.Probe(het_mem.out_conn, 'weights', synapse=None, sample_every=0.01)

sim = nengo.Simulator(model, dt=dt)
sim.run(run_time)

if plot_res:
    import matplotlib.pyplot as plt

    # figure out how to put these into a subplot
    plt.figure()
    plt.title("Error")
    plt.plot(np.linalg.norm(sim.data[p_error], axis=1))

    plt.figure()
    plt.title("Result")
    plt.plot(spa.similarity(sim.data[p_recall], vocab))
    plt.legend(vocab.keys, loc='best')
    plt.ylim(-0.5, 0.5)

    plt.figure()
    plt.title("Actual Answer")
    plt.plot(spa.similarity(sim.data[p_values], vocab))
    plt.ylim(-1.5, 1.5)

    plt.show()

ipdb.set_trace()
# I should make a wrapper for doing this quickly
base_name = "pred"
np.savez_compressed("data/%s_learning_data" % base_name, p_keys=sim.data[p_keys], p_recall=sim.data[p_recall],
                    p_error=sim.data[p_error], p_weights=sim.data[p_weights], p_values=sim.data[p_values])
np.savez_compressed("data/%s_learning_vocab" % base_name, keys=vocab.keys, vecs=vocab.vectors)

# no voja, single concatenated vector input
# used for making plot showing decoders continually switching back and forth

# make sure it messes up in the way you expect
# get the decoder data
# try to show the decoder values changing
# - look for decoders who's values change with every question
import itertools

from utils import gen_env_list, gen_vocab, add_to_env_list
from hetero_mem import build_hetero_mem, encoders, rebuild_hetero_mem
from constants import n_neurons, D, dt

import nengo
from nengo import spa
import numpy as np

import ipdb

plot_res = False

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

test_num = len(gtest_ans) + len(ftest_ans)
train_mult = 3
train_time = 1.0
run_time = 1.0


def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period/dt))
    if i_every != period/dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def f(t):
        i = int(round((t - dt)/dt))  # t starts at dt
        return x[(i/i_every) % len(x)]

    return f


run_num = 25
sample_every = 0.01
step_num = int(run_time / sample_every)
key_res = np.zeros((run_num, step_num, D*2))
val_res = np.zeros((run_num, step_num, D))
error_res = np.zeros((run_num, int(run_time / dt)))
recall_res = np.zeros((run_num, step_num, D))

# train the network and save the weights
with nengo.Network(label="Fast Net", seed=0) as train_model:
    env_key = nengo.Node(q_norm_list[0])
    env_value = nengo.Node([0]*10)

    recall = nengo.Node(size_in=D)
    learning = nengo.Node([0])

    # Generate hetero mem
    K = 400
    # This is usually calculated
    c = 0.51
    e = encoders(np.array(q_norm_list), K, rng)
    train_model.het_mem = build_hetero_mem(D * 2, D, e, c, pes_rate=pes_rate, voja_rate=voja_rate)

    nengo.Connection(env_key, train_model.het_mem.input, synapse=None)
    nengo.Connection(train_model.het_mem.output, recall)

    # Create the error population
    error = nengo.Ensemble(n_neurons * 8, D)
    nengo.Connection(learning, error.neurons, transform=[[10.0]] * n_neurons * 8,
                     synapse=None)
    nengo.Connection(learning, train_model.het_mem.in_conn.learning_rule, synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(env_value, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, train_model.het_mem.out_conn.learning_rule)

    # Setup probes
    p_enc = nengo.Probe(train_model.het_mem.in_conn, 'weights', synapse=None, sample_every=sample_every)
    p_dec = nengo.Probe(train_model.het_mem.out_conn, 'weights', synapse=None, sample_every=sample_every)

pre_enc = np.copy(train_model.het_mem.ens.encoders)
# simulate network
with nengo.Simulator(train_model) as train_sim:
    train_sim.run(0.5)

post_enc = np.copy(train_model.het_mem.ens.encoders)

ipdb.set_trace()

# with the saved weights, run a bunch of tiny simulations to check learning confidence
for seed_val, t_n in itertools.product(range(0, run_num), test_num):
    with nengo.Network(label="Fast Net", seed=seed_val) as model:
        if t_n < len(gtest_ans):
            env_key = nengo.Node(gtest_qs_nrm[t_n])
            env_value = nengo.Node(gtest_ans[t_n])
        else:
            env_key = nengo.Node(ftest_qs_nrm[t_n])
            env_value = nengo.Node(ftest_ans[t_n])

        recall = nengo.Node(size_in=D)
        learning = nengo.Node([0])

        # Generate hetero mem
        K = 400
        # This is usually calculated
        c = 0.51
        e = encoders(np.array(q_norm_list), K, rng)
        het_mem = rebuild_hetero_mem(input_w, output_w, train_model.het_mem.ens, pes_rate=pes_rate, voja_rate=voja_rate)

        nengo.Connection(env_key, het_mem.input, synapse=None)
        nengo.Connection(het_mem.output, recall)

        # Create the error population
        error = nengo.Ensemble(n_neurons*8, D)
        nengo.Connection(learning, error.neurons, transform=[[10.0]]*n_neurons*8,
                         synapse=None)
        nengo.Connection(learning, het_mem.in_conn.learning_rule, synapse=None)

        # Calculate the error and use it to drive the PES rule
        nengo.Connection(env_value, error, transform=-1, synapse=None)
        nengo.Connection(recall, error, synapse=None)
        nengo.Connection(error, het_mem.out_conn.learning_rule)

        # Setup probes
        p_keys = nengo.Probe(env_key, synapse=None, sample_every=sample_every)
        p_values = nengo.Probe(env_value, synapse=None, sample_every=sample_every)
        p_error = nengo.Probe(error, synapse=0.01)
        p_recall = nengo.Probe(recall, synapse=None, sample_every=sample_every)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(run_time)

    if t_n < len(gtest_ans):
        # save the values in a good way
        key_res[seed_val] = sim.data[p_keys]
        val_res[seed_val] = sim.data[p_values]
        error_res[seed_val] = np.sum(np.abs(sim.data[p_error]), axis=1)
        recall_res[seed_val] = sim.data[p_recall]


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
    plt.ylim(-0.5, 1.1)

    plt.figure()
    plt.title("Actual Answer")
    plt.plot(spa.similarity(sim.data[p_values], vocab))
    plt.ylim(-0.5, 1.1)

    plt.show()

ipdb.set_trace()
# I should make a wrapper for doing this quickly
base_name = "multpred2"
np.savez_compressed("data/%s_learning_data" % base_name, p_keys=key_res, p_recall=recall_res, p_error=error_res, p_values=val_res)
np.savez_compressed("data/%s_learning_vocab" % base_name, keys=vocab.keys, vecs=vocab.vectors)

# no voja, single concatenated vector input
# used for making plot showing decoders continually switching back and forth

# make sure it messes up in the way you expect
# get the decoder data
# try to show the decoder values changing
# - look for decoders who's values change with every question
import itertools
import datetime

from utils import gen_env_list, gen_vocab, add_to_env_list
from hetero_mem import build_hetero_mem, encoders, rebuild_hetero_mem
from constants import n_neurons, D, dt

import nengo
from nengo import spa
import numpy as np
import pandas as pd

pd_columns = ["key", "val", "error", "confidence"]
pd_res = []

# Learning rates
pes_rate = 0.01
voja_rate = 0.003

# Generate the vocab
rng = np.random.RandomState(0)
number_dict = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
               "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9}

max_sum = 9
max_num = max_sum - 2

number_list, vocab = gen_vocab(number_dict, max_num, D, rng)

join_num = "+".join(number_list[0:max_num])

# Create inputs and expected outputs
q_list, q_norm_list, ans_list = gen_env_list(number_dict, number_list, vocab, max_sum)


def sp_text(sp, vo):
    return vo.text(sp).split(';')[0].split(".")[1][2:]


def get_q_text(q_vec, voc):
    return "%s+%s" % (number_dict[sp_text(q_vec[:D], voc)], number_dict[sp_text(q_vec[D:], voc)])


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

# awkward shuffling for question order
ftest_qs_nrm[-1], ftest_qs_nrm[0] = ftest_qs_nrm[0], ftest_qs_nrm[-1]
ftest_ans[-1], ftest_ans[0] = ftest_ans[0], ftest_ans[-1]

test_num = len(gtest_ans) + len(ftest_ans)
train_mult = 3

run_num = 10


class SimpleEnv(object):

    def __init__(self, keys, values, env_period=0.1):
        self.keys = keys
        self.values = values
        self.env_idx = np.arange(len(keys))
        self.idx = 0
        self.shuffled = False
        self.i_every = int(round(env_period/dt))
        if self.i_every != env_period/dt:
            raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def get_key(self, t):
        return self.keys[self.idx]

    def get_val(self, t):
        return self.values[self.idx]

    def step(self, t):
        i = int(round((t - dt)/dt))  # t starts at dt
        ix = (i/self.i_every) % len(self.keys)
        if ix == 0 and not self.shuffled:
            print("shuffling")
            np.random.shuffle(self.env_idx)
            self.shuffled = True
        elif ix == 1:
            self.shuffled = False
        self.idx = self.env_idx(ix)

sample_every = 0.01
s_env = SimpleEnv(train_qs_nrm, train_ans, env_period=period)

# train the network and save the weights
with nengo.Network(seed=0) as train_model:

    env_key = nengo.Node(s_env.get_key)
    env_value = nengo.Node(s_env.get_val)

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
    p_enc = nengo.Probe(train_model.het_mem.in_conn.learning_rule, 'scaled_encoders', synapse=None,
                        sample_every=sample_every)
    p_dec = nengo.Probe(train_model.het_mem.out_conn, 'weights', synapse=None, sample_every=sample_every)

pre_enc = np.copy(train_model.het_mem.ens.encoders)
# simulate network
with nengo.Simulator(train_model) as train_sim:
    train_sim.run(period * train_mult * len(train_qs_nrm))

input_w = train_sim.data[p_enc][-1]
output_w = train_sim.data[p_dec][-1]

# with the saved weights, run a bunch of tiny simulations to check learning confidence
for seed_val, t_n in itertools.product(range(0, run_num), range(test_num)):
    with nengo.Network(label="Fast Net", seed=seed_val) as model:
        if t_n < len(gtest_ans):
            env_key = nengo.Node(gtest_qs_nrm[t_n])
            env_value = nengo.Node(gtest_ans[t_n])
        else:
            tmp_idx = t_n - len(gtest_ans)
            env_key = nengo.Node(ftest_qs_nrm[tmp_idx])
            env_value = nengo.Node(ftest_ans[tmp_idx])

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
        p_out = nengo.Probe(het_mem.output, synapse=0.01, sample_every=sample_every)
        p_recall = nengo.Probe(recall, synapse=None, sample_every=sample_every)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(period)

    pd_res.append([
        get_q_text(sim.data[p_keys][-1], vocab),
        number_dict[sp_text(sim.data[p_values][-1], vocab)],
        np.sum(np.abs(sim.data[p_error]), axis=1)[-1],
        np.max(spa.similarity(sim.data[p_recall], vocab))
    ])
    print("Finished run %s of test %s" % (seed_val, t_n))

# Save as Pandas dataframe
base_name = "multpred2"
df = pd.DataFrame(pd_res, columns=pd_columns)
hdf = pd.HDFStore("results/%s_%s.h5" % (base_name, datetime.datetime.now().strftime("%I_%M_%S")))
df.to_hdf(hdf, base_name)


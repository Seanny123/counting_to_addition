# I honestly have no idea what this is testing.

import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools
from random import shuffle
import ipdb

D = 32
dt = 0.001
run_time = 0.7 * 4 * 4

def create_adder_env(q_list, ans_list):
    with nengo.Network(label="env") as env:
        env.env_cls = AdderEnv(q_list, ans_list)
        env.timery = nengo.Node(env.env_cls.timey_things)

        env.env_values = nengo.Node(env.env_cls.fake_answer)
        env.env_keys = nengo.Node(env.env_cls.input_func)
        env.learning = nengo.Node(env.env_cls.learning_func)
    return env


class AdderEnv():

    def __init__(self, q_list, ans_list):
        ## Bunch of time constants
        self.rest = 0.05
        self.ans_delay = 0.35
        self.period = 0.7
        # each item gets run 4 times
        self.swap_time = self.period * 4

        self.list_index = 0
        self.q_list = q_list
        self.ans_list = ans_list
        self.num_items = len(q_list)

        # for confidence test only
        self.current_qs = self.q_list[0:2]
        self.current_ans = self.ans_list[0:2]
        self.swapped = False

    def input_func(self, t):
        if t % self.period > self.rest:
            return self.current_qs[self.list_index]
        else:
            return np.zeros(2*D)

    def fake_answer(self, t):
        # this feedback should be triggered on an answer
        if t % self.period > self.ans_delay:
            return self.current_ans[self.list_index]
        else:
            return np.zeros(D)

    def timey_things(self, t):
        "so I don't need to make multiple nodes"
        self.swap(t)
        self.increment_index(t)

    def swap(self, t):
        "for confidence test only, switch to shuffling later"
        i = int(round((t - dt)/dt))
        i_every = int(round(self.swap_time/dt))
        self.swapped = i/i_every % 2
        if self.swapped == 0:
            self.current_qs = self.q_list[0:2]
            self.current_ans = self.ans_list[0:2]
        else:
            self.current_qs = self.q_list[2:4]
            self.current_ans = self.ans_list[2:4]

    def increment_index(self, t):
        "instead of caluclating it in every function"
        i = int(round((t - dt)/dt))
        i_every = int(round(self.period/dt))
        self.list_index = i/i_every % len(self.current_qs)

    def learning_func(self, t):
        # this should be triggered on an answer
        # and then turned off after a certain point
        if t % self.period > self.ans_delay:
            return int(t <= self.swap_time*48)
        else:
            return 0

    def answer_checker(self, t, x):
        "check to see if the circuit has malfunctioned"
        return True

rng = np.random.RandomState(0)
vocab = spa.Vocabulary(D, unitary=["ONE"], rng=rng)
number_dict = {"ONE":1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5,
               "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
number_ordered = OrderedDict(sorted(number_dict.items(), key=lambda t: t[1]))

number_range = 4
number_list = number_ordered.keys()
for i in range(number_range):
    print(number_list[i])
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

n_neurons = 100

q_list = []
ans_list = []
for val in itertools.product(number_list, number_list):
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

with spa.SPA(vocabs=[vocab], label="Count Net", seed=0) as model:
    model.env = create_adder_env(q_list, ans_list)

    model.answer = spa.State(D)
    model.speech = spa.State(D)
    nengo.Connection(model.env.env_values, model.answer.input, synapse=None)

    adder = nengo.networks.AssociativeMemory(input_vectors=q_list, n_neurons=n_neurons)
    adder.add_wta_network()
    
    adder_out = nengo.Ensemble(n_neurons*8, len(ans_list))
    nengo.Connection(adder.elem_output, adder_out)
    conf_in = nengo.Ensemble(n_neurons*8, len(q_list))
    nengo.Connection(adder.elem_output, conf_in)

    model.recall = spa.State(D)
    model.conf = spa.State(1)

    nengo.Connection(model.env.env_keys, adder.input)

    conn_out = nengo.Connection(adder_out, model.recall.input, learning_rule_type=nengo.PES(1e-5),
                                function=lambda x: np.zeros(D))

    # Create the error population and node
    # These will come from the environment
    error = nengo.Ensemble(n_neurons*8, D)
    nengo.Connection(model.env.learning, error.neurons, transform=[[10.0]]*n_neurons*8,
                     synapse=None)

    def err_func(t, x):
        mag = np.linalg.norm(x[D])
        if mag < 0.1:
            return -1*x[-1]
        else:
            return 1*x[-1]

    err_mag = nengo.Node(err_func, size_in=D+1)
    nengo.Connection(error, err_mag[:D], synapse=0.01)
    nengo.Connection(model.env.learning, err_mag[-1])

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(model.env.env_values, error, transform=-1, synapse=None)
    nengo.Connection(model.recall.output, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Let confidence be inversely proportional to the magnitude of the error
    conn_conf = nengo.Connection(conf_in, model.conf.input, learning_rule_type=nengo.PES(1e-5),
                                 function=lambda x: 0.2)
    nengo.Connection(err_mag, conn_conf.learning_rule)


    feedback_actions = spa.Actions(
        fast="conf --> speech = recall",
        slow="1 - conf --> speech = answer"
    )
    model.feedback_bg = spa.BasalGanglia(feedback_actions)
    model.feedback_thal = spa.Thalamus(model.feedback_bg)
"""
    # Setup probes
    p_keys = nengo.Probe(model.env.env_keys, synapse=None)
    p_values = nengo.Probe(model.env.env_values, synapse=None)
    p_learning = nengo.Probe(model.env.learning, synapse=None)
    p_error = nengo.Probe(error, synapse=0.005)
    p_error_mag = nengo.Probe(err_mag, synapse=0.01)
    p_recall = nengo.Probe(model.recall.output, synapse=None)
    p_conf = nengo.Probe(model.conf.output)
    p_speech = nengo.Probe(model.speech.output)
    p_bg_out = nengo.Probe(model.feedback_bg.output)

sim = nengo.Simulator(model, dt=dt)
sim.run(run_time)

t = sim.trange()

import matplotlib.pyplot as plt

plt.figure()
plt.title("Error")
plt.plot(t, np.linalg.norm(sim.data[p_error], axis=1))

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)

ax1.title("Keys_1")
ax1.plot(t, spa.similarity(sim.data[p_keys][:, :D], vocab))
ax1.legend(vocab.keys, loc='best')
ax1.set_ylim([-1.25, 1.25])

ax2.title("Keys_2")
ax2.plot(t, spa.similarity(sim.data[p_keys][:, D:], vocab))
ax2.legend(vocab.keys, loc='best')
ax2.set_ylim([-1.25, 1.25])

ax3.title("Result")
ax3.plot(t, spa.similarity(sim.data[p_recall], vocab))
ax3.legend(vocab.keys, loc='best')
ax3.set_ylim([-1.25, 1.25])

ax4.title("Acutal Answer")
ax4.plot(t, spa.similarity(sim.data[p_values], vocab))
ax4.legend(vocab.keys, loc='best')
ax4.set_ylim([-1.25, 1.25])

plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

win = -400*18
plt.figure()
plt.title("Result")
plt.plot(t[win:], spa.similarity(sim.data[p_recall][win:,:], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Acutal Answer")
plt.plot(t[win:], spa.similarity(sim.data[p_values][win:,:], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Learning")
plt.plot(t[win:], sim.data[p_learning][win:,:])
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Keys_1")
plt.plot(t[win:], spa.similarity(sim.data[p_keys][win:, :D], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Keys_2")
plt.plot(t[win:], spa.similarity(sim.data[p_keys][win:, D:], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(t, sim.data[p_conf])
ax2.plot(t, sim.data[p_error_mag])
ax3.plot(t, sim.data[p_bg_out])
plt.show()

plt.plot(t, sim.data[p_add])
plt.show()

plt.show()

ipdb.set_trace()
"""
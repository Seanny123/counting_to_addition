import nengo
from nengo import spa

import numpy as np
from collections import OrderedDict
import itertools
from random import shuffle
import ipdb

D = 32

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
        self.period = 0.65
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
        "for confidence test only"
        if(round((t % self.swap_time) * 1000) == 0):
            if self.swapped:
                self.current_qs = self.q_list[0:2]
                self.current_ans = self.ans_list[0:2]
                self.swapped = False
            else:
                self.current_qs = self.q_list[2:4]
                self.current_ans = self.ans_list[2:4]
                self.swapped = True

    def increment_index(self, t):
        "instead of caluclating it in every function"
        if(round((t % self.period) * 1000) == 0):
            # This didn't work
            self.list_index = (self.list_index + 1) % len(self.current_qs)

    def learning_func(self, t):
        # this should be triggered on an answer
        # and then turned off after a certain point
        if t % self.period > self.ans_delay:
            return -int(t >= self.swap_time)
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
    adder_in = nengo.Ensemble(n_neurons*8, len(q_list))
    nengo.Connection(adder.elem_input, adder_in)

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
        mag = np.linalg.norm(x)
        if mag < 0.1:
            # TODO: find a better value or learning rate
            return 1
        else:
            return -mag

    err_mag = nengo.Node(err_func, size_in=D)
    nengo.Connection(error, err_mag)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(model.env.env_values, error, transform=-1, synapse=None)
    nengo.Connection(model.recall.output, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Let confidence be inversely proportional to the magnitude of the error?
    conn_conf = nengo.Connection(adder_in, model.conf.input, learning_rule_type=nengo.PES(1e-5),
                                 function=lambda x: 0.2)
    nengo.Connection(err_mag, conn_conf.learning_rule)


    feedback_actions = spa.Actions(
        fast="conf --> speech = recall",
        slow="1 - conf --> speech = answer"
    )
    model.feedback_bg = spa.BasalGanglia(feedback_actions)
    model.feedback_thal = spa.Thalamus(model.feedback_bg)

    # Setup probes
    p_keys = nengo.Probe(model.env.env_keys, synapse=None)
    p_values = nengo.Probe(model.env.env_values, synapse=None)
    p_learning = nengo.Probe(model.env.learning, synapse=None)
    p_error = nengo.Probe(error, synapse=0.005)
    p_error_mag = nengo.Probe(err_mag)
    p_recall = nengo.Probe(model.recall.output, synapse=None)
    p_conf = nengo.Probe(model.conf.output)

sim = nengo.Simulator(model)
sim.run(model.env.env_cls.swap_time*4)

t = sim.trange()

import matplotlib.pyplot as plt
"""
plt.figure()
plt.title("Error")
plt.plot(t, np.linalg.norm(sim.data[p_error], axis=1))

plt.figure()
plt.title("Keys_1")
plt.plot(t, spa.similarity(sim.data[p_keys][:, :D], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Keys_2")
plt.plot(t, spa.similarity(sim.data[p_keys][:, D:], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Result")
plt.plot(t, spa.similarity(sim.data[p_recall], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

plt.figure()
plt.title("Acutal Answer")
plt.plot(t, spa.similarity(sim.data[p_values], vocab))
plt.legend(vocab.keys, loc='best')
plt.ylim(-1.5, 1.5)

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

plt.plot(t, sim.data[p_conf])
plt.show()
plt.plot(t, sim.data[p_add])
plt.show()

plt.show()
"""
ipdb.set_trace()

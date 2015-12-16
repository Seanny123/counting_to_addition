from constants import *

import nengo
from nengo import spa
import numpy as np
import ipdb

from random import shuffle
import logging

logging.basicConfig(filename='env.log',level=logging.DEBUG)

# I'm pretty sure the gate is fucking up

def create_adder_env(q_list, ans_list, op_val, num_vocab):
    with nengo.Network(label="env") as env:
        env.env_cls = AdderEnv(q_list, ans_list, op_val, num_vocab)

        env.get_ans = nengo.Node(env.env_cls.get_answer)
        env.set_ans = nengo.Node(env.env_cls.set_answer, size_in=D)
        env.env_keys = nengo.Node(env.env_cls.input_func)

        env.op_in = nengo.Node(env.env_cls.op_state_input)
        env.q_in = nengo.Node(env.env_cls.q_inputs)
        env.learning = nengo.Node(lambda t: env.env_cls.learning)
        env.gate = nengo.Node(lambda t: env.env_cls.gate)
        env.reset = nengo.Node(lambda t: env.env_cls.chill)
        env.count_reset = nengo.Node(lambda t: -env.env_cls.learning)

    return env


class AdderEnv():

    def __init__(self, q_list, ans_list, op_val, num_vocab):
        ## Bunch of time constants
        self.rest = 0.05
        self.ans_duration = 0.3
        self.q_duration = 0.08
        self.op_duration = 0.05

        ## Value variables
        self.list_index = 0
        self.q_list = q_list
        self.ans_list = ans_list
        self.op_val = op_val
        self.num_items = len(q_list)
        self.indices = range(self.num_items)
        self.gate = 0
        self.num_vocab = num_vocab

        ## Timing variables
        self.learning = False
        self.ans_arrive = 0.0
        self.time = 0.0
        self.chill = False
        self.time_since_last_answer = 0.0
        self.questions_answered = 0

    def sp_text(self, x):
        return self.num_vocab.text(x).split(';')[0].split(".")[1][2:]

    def input_func(self, t):
        if self.time > self.rest:
            return self.q_list[self.indices[self.list_index]]
        else:
            return np.zeros(2*D)

    def q_inputs(self, t):
        if self.time > self.rest and self.time < (self.q_duration + self.rest):
            return self.q_list[self.indices[self.list_index]]
        else:
            return np.zeros(2*D)

    def op_state_input(self, t):
        if self.time > self.rest and self.time < (self.op_duration + self.rest):
            return self.op_val
        else:
            return np.zeros(less_D)

    def get_answer(self, t):
        if t < (self.ans_arrive + self.ans_duration) and self.ans_arrive != 0.0:
            return self.ans_list[self.indices[self.list_index]]
        else:
            return np.zeros(D)

    def set_answer(self, t, x):
        """Time keeping function.

        if there's some sort of answer coming from the basal-ganglia,
        detected by the norm not being (effectively) zero, give feedback for
        a certain amount of time before resetting the answer and starting the
        system again
        """
        self.time += dt
        self.time_since_last_answer += dt

        max_sim = np.max(self.num_vocab.dot(x))

        # while getting answer
        if max_sim > 0.3 and not self.chill:

            if self.ans_arrive == 0.0:
                self.gate = 1
                self.ans_arrive = t
                self.learning = True

                correct_text = self.sp_text(self.ans_list[self.indices[self.list_index]])
                ans_text = self.sp_text(x)
                self.time_since_last_answer = 0.0
                self.questions_answered += 1
                print("Questions answered %s" %self.questions_answered)
                if correct_text != ans_text:
                    logging.debug("%s != %s" %(correct_text, ans_text))
                    print("%s != %s" %(correct_text, ans_text))
                    ipdb.set_trace()

            elif t > (self.ans_arrive + self.ans_duration):
                self.ans_arrive = 0.0
                self.gate = 0
                self.chill = True
                self.learning = False

        elif max_sim < 0.2 and self.chill:
            
            if self.list_index < self.num_items - 1:
                self.list_index += 1
            else:
                shuffle(self.indices)
                self.list_index = 0


            self.chill = False
            self.time = 0.0

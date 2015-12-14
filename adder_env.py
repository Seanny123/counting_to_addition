from constants import *

import nengo
from nengo import spa
import numpy as np
import sys

from random import shuffle
import logging

logging.basicConfig(filename='env.log',level=logging.DEBUG)


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
    return env


class AdderEnv():

    def __init__(self, q_list, ans_list, op_val, num_vocab):
        ## Bunch of time constants
        self.rest = 0.05
        self.ans_duration = 0.3
        self.q_duration = 0.07
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

    def get_sp_text(self, x):
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
            return np.zeros(len(self.op_val))

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

        mag = np.linalg.norm(x)

        # while getting answer
        if mag > 0.3 and not self.chill:

            if self.ans_arrive == 0.0:
                print("ANS %s" %t)
                self.gate = 1
                self.ans_arrive = t
                self.learning = True

                correct_text = self.get_text(self.ans_list[self.indices[self.list_index]])
                ans_text = self.get_text(x)
                if correct_text != ans_texts:
                    logging.debug("%s != %s" %(correct_text, ans_text))

            elif t > (self.ans_arrive + self.ans_duration):
                print("chill: %s" %t)
                self.ans_arrive = 0.0
                self.gate = 0
                self.chill = True
                self.learning = False

        elif mag < 0.3 and self.chill:
            print("stop chilling: %s" %t)
            
            if self.list_index < self.num_items - 1:
                self.list_index += 1
                print("after increment: %s \n" %self.list_index)
            else:
                shuffle(self.indices)
                self.list_index = 0


            self.chill = False
            self.time = 0.0

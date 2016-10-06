from constants import *

import nengo
from nengo import spa
import numpy as np
import ipdb

from random import shuffle
import logging

logging.basicConfig(filename='env.log',level=logging.DEBUG)


def create_adder_env(q_list, q_norm_list, ans_list, op_val, num_vocab, ans_dur=0.3):
    with nengo.Network(label="env") as env:
        env.env_cls = AdderEnv(q_list, q_norm_list, ans_list, op_val, num_vocab, ans_dur)

        env.get_ans = nengo.Node(env.env_cls.get_answer)
        env.set_ans = nengo.Node(env.env_cls.set_answer, size_in=D)
        env.env_keys = nengo.Node(env.env_cls.input_func)
        env.env_norm_keys = nengo.Node(env.env_cls.input_func_normed)

        env.op_in = nengo.Node(env.env_cls.op_state_input)
        env.q_in = nengo.Node(env.env_cls.q_inputs)
        env.learning = nengo.Node(lambda t: env.env_cls.learning)
        env.count_reset = nengo.Node(lambda t: -env.env_cls.learning - 1)

    return env


class AdderEnv():

    def __init__(self, q_list, q_norm_list, ans_list, op_val, num_vocab, ans_dur, filename="paper5_reactions.txt"):
        ## Bunch of time constants
        self.rest = 0.05
        self.ans_duration = ans_dur
        self.q_duration = 0.08
        self.op_duration = 0.05

        ## Value variables
        self.list_index = 0
        self.q_list = q_list
        self.q_norm_list = q_norm_list
        self.ans_list = ans_list
        self.op_val = op_val
        self.num_items = len(q_list)
        self.indices = range(self.num_items)
        self.num_vocab = num_vocab

        ## Timing variables
        self.learning = -1
        self.ans_arrive = 0.0
        self.time = 0.0
        self.train = False
        self.reset = False
        # For measuring progress
        self.questions_answered = 0
        # For detecting a crash
        self.time_since_last_answer = 0.0

        # Logging for reaction times
        self.fi = open("data/%s" %filename, "w")
        self.react_time = open("data/paper5_react_time.txt", "w")

    def sp_text(self, x):
        return self.num_vocab.text(x).split(';')[0].split(".")[1][2:]

    # TODO: These functions should be combined as a closure
    def input_func(self, t):
        if self.time > self.rest and not self.reset:
            return self.q_list[self.indices[self.list_index]]
        else:
            return np.zeros(2*D)

    def input_func_normed(self, t):
        if self.time > self.rest and not self.reset:
            return self.q_norm_list[self.indices[self.list_index]]
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

        this is basically a temporally sensitive state machine, however
        I don't know of any state machine libraries for Python, so this is
        what you get instead...

        WHY DO I HAVE SUCH A HARD TIME WRITING STATE MACHINES
        """
        self.time += dt
        self.time_since_last_answer += dt

        max_sim = np.max(np.dot(self.num_vocab.vectors, x))

        # when an answer arrives, note it's time of arrival and turn on learning
        if max_sim > 0.45 and self.ans_arrive == 0.0 and not self.train:
            self.ans_arrive = t
            self.learning = 0
            self.train = True

            # check the answer is correct
            correct_text = self.sp_text(self.ans_list[self.indices[self.list_index]])
            ans_text = self.sp_text(x)
            self.react_time.write("%s\n" % self.time_since_last_answer)
            self.time_since_last_answer = 0.0
            self.questions_answered += 1
            q_ans = self.q_list[self.indices[self.list_index]]
            addend_1 = self.sp_text(q_ans[:D])
            addend_2 = self.sp_text(q_ans[D:])
            print("Answered %s+%s" % (addend_1, addend_2))
            print("Answered %s questions at %s" % (self.questions_answered, t))
            self.fi.write("Question answered %s at %s\n" % (self.questions_answered, t))
            self.fi.write("max_sim: %s\n" % max_sim)
            if correct_text != ans_text:
                logging.debug("%s != %s" %(correct_text, ans_text))
                print("%s != %s" %(correct_text, ans_text))
                self.fi.write("Error: %s\n" %t)
                # This should just change the learning, not totally stop the simulation
                #ipdb.set_trace()

        # sustain the answer for training purposes

        # after we're done sustaining the answer
        # turn of the learning and the answer arrival time
        # wait until the similarity goes down before asking for a new question
        if t > (self.ans_arrive + self.ans_duration) and self.train:
            if not self.reset:
                self.ans_arrive = 0.0
                self.learning = -1
                self.reset = True
                self.fi.write("Turning off: %s\n" %t)

            if max_sim < 0.1:
                self.fi.write("Next question: %s\n" %t)
                self.fi.write("max_sim: %s\n" %max_sim)
                self.time = 0.0
                self.train = False
                self.reset = False

                if self.list_index < self.num_items - 1:
                    self.list_index += 1
                    print("Increment:: %s" %self.list_index)
                else:
                    print("Shuffling\n")
                    self.fi.write("Shuffling\n")
                    shuffle(self.indices)
                    self.list_index = 0

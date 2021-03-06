from nengo import spa
import nengo
from scipy.linalg import sqrtm
import numpy as np

from collections import OrderedDict
import itertools


def nearest(d, rng=np.random):
    p = nengo.dists.UniformHypersphere(surface=True).sample(d, d, rng=rng)
    return np.dot(p, np.linalg.inv(sqrtm(np.dot(p.T, p))))


def gen_vocab(n_dict, n_range=9, dims=32, rng=None):

    vo = spa.Vocabulary(dims, rng=rng)

    if rng is None:
        rng = np.random
    orth_vecs = nearest(dims, rng=rng)

    number_ordered = OrderedDict(sorted(n_dict.items(), key=lambda t: t[1]))
    n_list = list(number_ordered.keys())

    for i in range(n_range):
        print(n_list[i])
        vo.add(n_list[i], orth_vecs[i])

    return n_list, vo


def add_to_env_list(val, ans_val, q_list, q_norm_list, ans_list, number_list, vocab):
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
        vocab.parse(number_list[ans_val - 1]).v
    )


def gen_env_list(number_dict, number_list, vocab, max_sum):
    q_list = []
    q_norm_list = []
    ans_list = []

    for val in itertools.product(number_list, number_list):
        # Filter for min count
        if val[0] <= val[1]:
            ans_val = number_dict[val[0]] + number_dict[val[1]]
            if ans_val <= max_sum:
                add_to_env_list(val, ans_val, q_list, q_norm_list, ans_list, number_list, vocab)
                print("%s+%s=%s" % (val[0], val[1], number_list[ans_val-1]))

    return q_list, q_norm_list, ans_list

import nengo
import numpy as np
import ipdb

import matplotlib.pyplot as plt

plt.plot(np.sum(np.abs(sim.data[p_error]), axis=1))

count_res = sim.data[p_count_res]
count_tot = sim.data[p_count_tot]
count_fin = sim.data[p_count_fin]
t_count = np.arange(count_res.shape[0]) * 0.005

s_win = 0
win = -1

# basic count plots

fig, (p1, p2, p3) = plt.subplots(3, sharex=True, figsize=(20,10))

p1.plot(t_count[s_win:win], spa.similarity(count_res, vocab)[s_win:win])
p1.legend(vocab.keys, bbox_to_anchor=(1.1, 0), frameon=True, fancybox=True, shadow=True)
p1.set_ylabel("Count Result")

p2.plot(t_count[s_win:win], spa.similarity(count_tot, vocab)[s_win:win])
p2.set_ylabel("Times Counted")

p3.plot(t_count[s_win:win], spa.similarity(count_fin, vocab)[s_win:win])
p3.set_ylabel("Times to Count")
p3.set_xlabel("Time (s)")

plt.show()


bg_in = sim.data[p_bg_in]
bg_out = sim.data[p_bg_out]

final_ans = sim.data[p_final_ans]
speech = sim.data[p_speech]

ans_assoc = sim.data[p_ans_assoc]
thres_val = sim.data[p_thres_ens]

t_fin = np.arange(final_ans.shape[0]) * 0.01

fig, (p1, p2, p3, p4) = plt.subplots(4, sharex=True, figsize=(20,10))

p1.plot(t_fin[s_win:win], thres_val[s_win:win])
p1.legend(vocab.keys, bbox_to_anchor=(1.1, 0), frameon=True, fancybox=True, shadow=True)
p1.set_ylabel("Threshold")

p2.plot(t_fin[s_win:win], spa.similarity(ans_assoc, vocab)[s_win:win])
p2.set_ylabel("Answer")

p3.plot(t_fin[s_win:win], spa.similarity(final_ans, vocab)[s_win:win])
p3.set_ylabel("Final Answer")

p4.plot(t_fin[s_win:win], spa.similarity(speech, vocab)[s_win:win])
p4.set_ylabel("Speech")
p4.set_xlabel("Time (s)")

plt.show()

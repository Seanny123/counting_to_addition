import matplotlib.pyplot as plt

trange = sim.trange()

plt.plot(trange, sim.data[p_err])
plt.show()
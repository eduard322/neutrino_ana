import numpy as np
import matplotlib.pyplot as plt

E0, E1 = 10, 10
a, b = 0.1, 0.15

E0_list = np.array([E0]*10000)
E1_list = np.array([E1]*10000)

apply_smearing = np.vectorize(np.random.normal)


E0_list = apply_smearing(E0_list, a*E0_list)
E1_list = apply_smearing(E1_list, b*E1_list)

fig, ax = plt.subplots(figsize = (6,6))

ax.hist(np.abs(E0_list - E1_list), bins = 50, histtype = "step")

plt.show()
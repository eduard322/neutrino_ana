import numpy as np
import matplotlib.pyplot as plt


E0_list = np.array([10]*10000)

apply_smearing = np.vectorize(np.random.normal)


E0_list = apply_smearing(E0_list, 0.1*E0_list)

fig, ax = plt.subplots(figsize = (6,6))

ax.hist(E0_list, bins = 50, histtype = "step")

plt.show()
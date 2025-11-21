import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("timing.dat")


plt.clf()
plt.plot(data["t"], data["pde"] / data["nn"], linewidth=2)
# plt.legend(loc="upper left")

ax = plt.gca()
ax.set_xlabel("T")
ax.set_ylabel("speedup")
ax.set_xlim(0.5, 3.14)
ax.set_ylim(4, 10)
plt.savefig("runtime.pdf", bbox_inches="tight")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

lowest = 0      # start of full time
highest = 1000  # end of full time
t_interval = 50 # E(time interval) = 50 

start = np.random.randint(lowest, highest)
mu, sigma = start + t_interval, 0.2 *t_interval # values for the normal distribution
end = truncnorm((start - mu) / sigma, (highest - mu) / sigma, loc=mu, scale=sigma)

print(end.rvs(1))
print(round(end.rvs(1)[0]))

fig, ax = plt.subplots(1)
ax.hist(end.rvs(10000))
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.dates as mdates
path_to_output = '~/internship/output'

nt = np.array([1,2,4,8,16])
error = np.array([0.192149937915012, 0.17499162599994, 0.158571830313516, 0.155529167118429, 0.151117539784669])
time = ['00:12:46', '00:17:08', '00:28:35', '00:51:26', '01:55:45']
time_y = [datetime.strptime(item, '%H:%M:%S') for item in time]


fig1, ax1 = plt.subplots()
ax1.set_ylim([0, 0.2])
ax1.plot(nt, error, marker='o')
ax1.set(xlabel='Number of timesteps', ylabel=r'Normalized $L^2$ loss',
        title='Error after 1000 epochs')
ax1.grid()
fig1.tight_layout()
fig1.savefig(f'{path_to_output}/timestep.png')

fig2, ax2 = plt.subplots()
ax2.plot(nt, time_y, marker='o')
ax2.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S',))
ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax2.set(xlabel='Number of timesteps', ylabel='Runtime',
        title='Runtime for 1000 epochs')
ax2.grid()
fig2.tight_layout()
fig2.savefig(f'{path_to_output}/runtime.png')
plt.close()
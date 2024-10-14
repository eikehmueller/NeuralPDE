import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.dates as mdates
path_to_output = '~/internship/output'

# timesteps data
nt = np.array([1,2,4,8,16])
error_t = np.array([0.192149937915012, 0.17499162599994, 0.158571830313516, 0.155529167118429, 0.151117539784669])
time_t = ['00:12:46', '00:17:08', '00:28:35', '00:51:26', '01:55:45']
time_y = [datetime.strptime(item, '%H:%M:%S') for item in time_t]

# dynamic variables data
n_dyn = np.array([7, 10, 12, 14, 16, 18])
error_d = np.array([0.158571830313516,0.106692006212527, 0.0871888816413165, 0.071316738788293, 0.0554150781583446, 0.0466916302361248])
time_d = ['00:28:35', '00:48:15', '00:54:21', '00:59:04', '01:06:43', '01:09:36']
time_dy = [datetime.strptime(item, '%H:%M:%S') for item in time_d]


# barchart data
labels = ['Base case', r'2 $\times$ batchsize', '+3 dyn', '+3 ancil', '+0.0002 lr', '+1 mesh ref', r'+1 mesh ref, 2 $\times$ nt', '+1 radial ref', '+1 dual ref']
error_values = np.array([0.158571830313516, 0.158144524372999, 0.106692006212527, 0.162846121462574, 0.151727263336801, 0.476709717655648, 0.473258504935402, 0.106020766214592, 0.308887728538425])
time_values = ['00:28:35', '01:12:23', '00:48:15', '00:50:53', '00:28:34', '00:28:55', '00:51:28', '00:48:42', '03:13:21']

# timesteps vs error
fig1, ax1 = plt.subplots()
ax1.set_ylim([0, 0.2])
ax1.plot(nt, error_t, marker='o')
ax1.set(xlabel='Number of timesteps', ylabel=r'Normalized $L^2$ loss',
        title='Error after 1000 epochs')
ax1.grid()
fig1.tight_layout()
fig1.savefig(f'{path_to_output}/timestep.png')

#timesteps vs runtime
fig2, ax2 = plt.subplots()
ax2.plot(nt, time_y, marker='o')
ax2.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S',))
ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax2.set(xlabel='Number of timesteps', ylabel='Runtime',
        title='Runtime for 1000 epochs')
ax2.grid()
fig2.tight_layout()
fig2.savefig(f'{path_to_output}/runtime.png')

# dynamic variables vs error
fig3, ax3 = plt.subplots()
ax3.set_ylim([0, 0.2])
ax3.plot(n_dyn, error_d, marker='o')
ax3.set(xlabel='Number of dynamic variables', ylabel=r'Normalized $L^2$ loss',
        title='Error after 1000 epochs')
ax3.grid()
fig3.tight_layout()
fig3.savefig(f'{path_to_output}/dynamic.png')

# dynamic variables vs runtime
fig4, ax4 = plt.subplots()
ax4.plot(n_dyn, time_dy, marker='o')
ax4.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S',))
ax4.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax4.set(xlabel='Number of dynamic variables', ylabel='Runtime',
        title='Runtime for 1000 epochs')
ax4.grid()
fig4.tight_layout()
fig4.savefig(f'{path_to_output}/dynamic_runtime.png')




# barchart changing hyperparameters, error and runtime
fig5, ax5 = plt.subplots()
ax5.bar(labels, error_values)
fig5.savefig(f'{path_to_output}/hyperparams.png')
plt.close()


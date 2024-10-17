# module to draw plots found in the report and the presentations. Data is taken from the excel sheet Numerical Tests.

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.dates as mdates

import matplotlib.ticker as ticker 

path_to_output = '../output'

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

# radial mesh data
n_rad = np.array([18,36,60])
error_r = np.array([0.158571830313516, 0.106020766214592, 0.0739858624173744])
time_r = ['00:28:35', '00:48:42', '00:45:00']
time_ry = [datetime.strptime(item, '%H:%M:%S') for item in time_r]

# barchart data
labels = ['Original', r'2 $\times \ nt$', r'0.5 $\times \ nt$', r'2 $\times \ batchsize$', r'+3 $n_{dyn}$', r'+3 $n_{ancil}$', '+0.0002 $lr$', r'+1 $n_{ref}$', r'+1 $n_{ref}, \  2 \times nt$', r'more $n_{points}$', r'+1 $n_{dual-ref}$']
parameter_data = {
    'L2 error': (0.158571830313516, 0.155529167118429, 0.17499162599994, 0.158144524372999, 0.106692006212527, 0.162846121462574, 0.151727263336801, 0.476709717655648, 0.473258504935402, 0.106020766214592, 0.308887728538425),
    'Runtime': (28.42, 51.43, 17.13, 72.38, 48.25, 50.88, 28.56 , 28.92, 51.46, 48.7, 193.35),
}

relative_parameter_data = {
    'L2 error': (0.0, -0.0030426631950869754, 0.016419795686424016, -0.00042730594051698656, -0.051879824100988986, 0.0042742911490580016, -0.006844566976714983, 0.31813788734213, 0.31468667462188604, -0.05255106409892399, 0.150315898224909),
    'Runtime': ( 0.0, 23.01, -11.29,  43.96,  19.83,  22.46, 0.14,  0.5,  23.04,  20.28,  164.93),
}
error_values = np.array([0.158571830313516, 0.155529167118429, 0.17499162599994, 0.158144524372999, 0.106692006212527, 0.162846121462574, 0.151727263336801, 0.476709717655648, 0.473258504935402, 0.106020766214592, 0.308887728538425])
#time_values = ['00:28:35', '00:51:26', '00:17:08', '01:12:23', '00:48:15', '00:50:53', '00:28:34', '00:28:55', '00:51:28', '00:48:42', '03:13:21']
#timevalues_transf = [datetime.strptime(item, '%H:%M:%S') for item in time_values]


x = np.arange(len(labels))
width = 0.4
multiplier = 0

time_values = np.array([28.42, 51.43, 17.13, 72.38, 48.25, 50.88, 28.56 , 28.92, 51.46, 48.7, 193.35])
time_relative = time_values - 28.42
error_relative = list(error_values - 0.158571830313516)

#time_relative = ['00:00:00', '00:22:51', '00:11:27', '00:43:48', '00:19:40', '00:22:18', '00:28:34', '00:00:01', '00:22:53', '00:20:07', '02:44:46']

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

# points in patch vs error
fig5, ax5 = plt.subplots()
ax5.set_ylim([0, 0.2])
ax5.plot(n_rad, error_r, marker='o')
ax5.set(xlabel='Number of points on radius', ylabel=r'Normalized $L^2$ loss',
        title='Error after 1000 epochs')
ax5.grid()
fig5.tight_layout()
fig5.savefig(f'{path_to_output}/radial.png')

# points in patch vs runtime
fig6, ax6 = plt.subplots()
ax6.plot(n_rad, time_ry, marker='o')
ax6.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S',))
ax6.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax6.set(xlabel='Number of points on radius', ylabel='Runtime',
        title='Runtime for 1000 epochs')
ax6.xaxis.set_major_locator(ticker.AutoLocator())
ax6.grid()
fig6.tight_layout()
fig6.savefig(f'{path_to_output}/radial_runtime.png')



# barchart changing hyperparameters, error and runtime
fig7, ax7a = plt.subplots(figsize=(14,7))#
ax7b = ax7a.twinx()

for attribute, measurement in parameter_data.items():
    if attribute == 'L2 error':
        offset = width * multiplier
        rects1 = ax7a.bar(x + offset, measurement, width=0.4, label='L2 error', align='edge', color='tab:blue')
        ax7a.bar_label(rects1, padding=3)
        multiplier += 1
    else:
        offset = width * multiplier
        rects2 = ax7b.bar(x + 2 * offset, measurement, width=-0.4, label='Runtime', align='edge', color='tab:red')
        ax7b.bar_label(rects2, padding=3)
        multiplier += 1
ax7a.tick_params(axis='x', labelrotation=45, pad=10)
ax7a.set_ylabel('L2 error', color='tab:blue')
ax7b.set_ylabel('Runtime (minutes)', color='tab:red')
ax7a.set_xticks(x + width, labels)
ax7a.tick_params(axis='y', labelcolor='tab:blue')
ax7b.tick_params(axis='y', labelcolor='tab:red')
ax7a.set_title('Hyperparameters')
ax7a.legend((rects1, rects2), (rects1.get_label(), rects2.get_label()), loc='upper left')
fig7.tight_layout()
fig7.savefig(f'{path_to_output}/bar_chart_hyperparams.png')


#fig5b, ax5b = plt.subplots(figsize=(14,7))
#bar_container1b = ax5b.bar(labels, time_values, align='center', label='Runtime')
#ax5b.bar_label(bar_container1b)
#ax5b.set_ylabel('Runtime (minutes)')
#ax5b.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
#ax5b.tick_params(axis='y')
#ax5a.set_title('Change in hyperparameters')

#fig5b.tight_layout()
#fig5b.savefig(f'{path_to_output}/bar_chart_hyperparams_runtime.png')



plt.close()


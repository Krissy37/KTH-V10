# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:32:28 2024

@author: Kristin
"""
# =============================================================================
# example for L Shell and M Shell function in KTH V10b
# =============================================================================

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt, dates as mdates
from matplotlib.patches import Wedge

#import sys
#sys.path.append('C:\\Users\\Kristin\\Documents\\PhD\\KTH_V10\\') # adapt this local path!
import kth_model_for_mercury_v10b as kth
control_param_path = 'control_params_v10.json'
fit_param_path = 'kth_own_cf_fit_parameters_v10.dat'


import os
import sys


# # System Path to the KTH22-model-main folder
# your_System_Path = 'C:\\Users\\Kristin\\Documents\\PhD\\KTH_V10\KTH22_V10_ext_usr\\'


# sys.path.append(your_System_Path)
# import kth_model_for_mercury_v10 as kth
# control_param_path = os.path.join(your_System_Path, 'control_params_v10.json')
# fit_param_path = os.path.join(your_System_Path, 'kth_own_cf_fit_parameters_v10.dat')

# =============================================================================
# define contants
# =============================================================================

R_M = 2440 #in km, Mercury Radius 

x_start = np.array([1.2 ])* R_M
y_start = np.array([0.0])
z_start = np.array([0.3])*R_M

r_hel = np.array([0.39])
di = np.array([50])

aberration = 0
imf_bx = imf_by = imf_bz = 0


# =============================================================================
# calculate L Shell parameter
# =============================================================================

l_shell = kth.calc_L_shell_v10(x_start, y_start, z_start, r_hel, di, aberration, control_param_path, fit_param_path)

print('L Shell: ', l_shell)


# =============================================================================
# calculate M Shell parameter
# =============================================================================

m_shell = kth.calc_M_shell_v10(x_start, y_start, z_start, r_hel, di, aberration, control_param_path, fit_param_path)
r_m_shell = np.round(np.sqrt(m_shell[0]**2 + m_shell[1]**2 + m_shell[2]**2), 4)/R_M
    
print('Position of minimal magnetic field magnitude along fieldline: ', m_shell/R_M)
print('Radius: ', r_m_shell)


# # =============================================================================
# # plot 
# # =============================================================================

fieldline_a = kth.trace_fieldline_v10(x_start, y_start, z_start, r_hel, di, aberration, control_param_path, fit_param_path)
fieldline_b = kth.trace_fieldline_v10(x_start, y_start, z_start, r_hel, di, aberration, control_param_path, fit_param_path, delta_t_in= -0.9)

    
theta1, theta2 = 90, 90 + 180
radius = 1
center = (0, 0)
m_day1 = Wedge(center, radius, theta1, theta2, fc='0.4', edgecolor='0.4')
m_night1 = Wedge(center, radius, theta2, theta1, fc='0.8', edgecolor='0.8')


lim = 3
title = 'fieldlinetrace'

fig = plt.figure(figsize=(6, 6))
fig.suptitle(title, fontsize=16)

#x-z
ax = plt.subplot(1, 1, 1)
ax.axis('square')             # creates square ax
plt.xlim((lim,-lim))          # set axis limits
plt.ylim((-lim,lim))
plt.scatter(fieldline_a[0]/R_M, fieldline_a[2]/R_M, c = fieldline_a[3])
plt.scatter(fieldline_b[0]/R_M, fieldline_b[2]/R_M, c = fieldline_b[3])
plt.plot(m_shell[0][0][0]/R_M, m_shell[2][0][0]/R_M, 'o', color = 'red', label = 'minimum |B|')
ax.add_artist(m_day1)   
ax.add_artist(m_night1)                                                              # creats grey circle (mercury)     
plt.xlabel(r'$X_{MSM}$' +' in '+'$R_M$')
plt.ylabel(r'$Z_{MSM}$' +' in '+'$R_M$')
plt.grid()
plt.legend()
plt.colorbar()


B_total_nT_fieldline = np.round(np.concatenate((np.flip(fieldline_a[3].astype(float)), fieldline_b[3].astype(float))),5)

plt.figure(figsize=(8, 6))
plt.plot(B_total_nT_fieldline)
plt.title('|B| along Fieldline')
plt.ylabel('|B| in nT')
plt.xlabel('index')
plt.grid()
plt.show()


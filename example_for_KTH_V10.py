# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:37:25 2023

@author: Kristin
"""
# -*- coding: utf-8 -*-

#import sys

#sys.path.append('C:\\Users\\Kristin\\Documents\\PhD\\KTH_V10\\') # adapt this local path!
import kth_model_for_mercury_v10b as kth
control_param_path = 'control_params_v10.json'
fit_param_path = 'kth_own_cf_fit_parameters_v10.dat'
import numpy as np

# =============================================================================
# This is an example for how to run the KTH-Model to calculate the magnetic 
# field inside the hermean magnetosphere. 
# If you run this routine and get the output written below, everything works as it shoud. 
#
# Note: The model only calculates the magnetic field for points inside the magnetosphere. 
# Otherwise the output will be 'nan'. 
#
# If you want to change the magnetopause distance, this must be translated in a change in r_hel. 
# Only change the parameters in the function call. Don't change the parameter files. 
#
# There will be updates of parameters and modules. Status as of 17.05.2023. 
#
# If you have any questions, do not hesitate to cantact me (Kristin Pump, email: k.pump@tu-bs.de)
# =============================================================================


# radius of Mercury in km 
R_M = 2440

#define coordinates in mso
x_mso = np.linspace(1.0, 4.0, 20)*R_M
y_mso = np.zeros(len(x_mso))
z_mso = np.linspace(0.2, 1, 20)*R_M

imf_bx = imf_by = imf_bz = 0   #not ready to use yet. Set it to 0. 
aberration = 0   #aberration angle in degrees (angle should be negative)

#define heliocentric distance and disturbance index per data point
r_hel = np.ones(len(x_mso))* 0.38
di = np.ones(len(x_mso))*50


#run model 
B_KTH =  kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                        control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                        dipole=True, neutralsheet=True, 
                                        ringcurrent=True, internal=True, 
                                        external=True)

Bx_KTH = B_KTH[0]
By_KTH = B_KTH[1]
Bz_KTH = B_KTH[2]

#print input coordinates
print('x (in MSO in km): ', x_mso)
print('y (in MSO in km): ', y_mso)
print('z (in MSO in km): ', z_mso)
print('\n')

#print magnetic field output
print('Bx KTH in nT:', Bx_KTH)
print('By KTH in nT:', By_KTH)
print('Bz KTH in nT:', Bz_KTH)

# =============================================================================
# if the model works correctly it should print: 
    
# x (in MSO in km):  [-2440. -3050. -3660. -4270. -4880.]
# y (in MSO in km):  [-1952. -2074. -2196. -2318. -2440.]
# z (in MSO in km):  [-488. -244.    0.  244.  488.]


# Bx KTH in nT: [-65.41848183 -35.51500702 -19.27779438  -8.87159252   0.56248625]
# By KTH in nT: [-44.9178099  -18.18511262  -6.56756208  -1.79356723   0.14939532]
# Bz KTH in nT: [70.75145241 50.91643894 31.11972314 17.86344264 11.98107602]
# =============================================================================


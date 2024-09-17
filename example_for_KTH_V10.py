# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:37:25 2023

@author: Kristin
"""
# -*- coding: utf-8 -*-

#import sys
#sys.path.append('C:\\ ... your path ... \\') # adapt this local path!

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
# There is an extra routine to calculate the subsolar standoff distance R_SS of the magnetopause (
# kth.calc_R_SS_km). Example in line 81.  
#
# There will be updates of parameters and modules. Status as of 17th September 2024. 
#
# If you have any questions, do not hesitate to cantact me (Kristin Pump, email: k.pump@tu-bs.de)
# =============================================================================


# radius of Mercury in km 
R_M = 2440

#define coordinates in mso in km
x_mso = np.linspace(1.0, -4.0, 5)*R_M
y_mso = np.zeros(len(x_mso))
z_mso = np.linspace(1.2, 1, 5)*R_M

imf_bx = imf_by = imf_bz = 0     #not ready to use yet. Set it to 0. 
aberration = 0                   #aberration angle in degrees (average ~ 7° )

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
# calc_R_SS_km is a new function in Version 10. This function 
# calculates the subsolar standoff distance for a given set of r_hel and di 
# (and the control parameter file as in KTH) in kilometers. 
# =============================================================================

R_SS = kth.calc_R_SS_km(r_hel, di, control_param_path)
print('\n')
print('new function in Version 10:')
print('R SS: ', R_SS)


# =============================================================================
# aberration estimator
# estimator for aberration angle for different heliocentric distances and
# different solar wind velocities. Default velocity is 400 km/s. 
# =============================================================================

print('\n')
r_hel = 0.37    #AU
v_sw = 400      #km/s

estimated_aberration = kth.estimate_aberration_angle(r_hel, v_sw )
print('estimated aberration angle: ', estimated_aberration, '°')


# =============================================================================
# if the model works correctly it should print: 
#    
#x (in MSO in km):  [ 2440.  -610. -3660. -6710. -9760.]
#y (in MSO in km):  [0. 0. 0. 0. 0.]
#z (in MSO in km):  [2928. 2806. 2684. 2562. 2440.]
#
#
#Bx KTH in nT: [-59.18937442 173.20287364  75.10562428  48.35518278  42.54176103]
#By KTH in nT: [ 0.00000000e+00 -1.81480346e-14 -6.11403696e-15 -8.91102109e-16 -2.14163130e-16]
#Bz KTH in nT: [  33.00925789 -342.71088344    8.01092884   11.05015624   25.81157564]


#new function in Version 10:
#R SS:  [3456.89781682 3456.89781682 3456.89781682 3456.89781682 3456.89781682]


#estimated aberration angle:  7.130861692415381

# =============================================================================





# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:47:11 2024

@author: Kristin
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:48:08 2024

@author: Kristin
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import kth_model_for_mercury_v10b as kth
control_param_path = 'control_params_v10.json'
fit_param_path = 'kth_own_cf_fit_parameters_v10.dat'
import numpy as np

# =============================================================================
# This routine is designed to check different functions of the KTH model (Version 10b) 
#
# Test 1: normal input --> checks wether magnetic field calculation is correct 
# Test 2: wrong input --> model should inform the user about wrong input
# =============================================================================



check_tests = np.array([False, False, False, False, False, False, False])

# =============================================================================
#  check KTH version
# =============================================================================

R_M = 2440

# =============================================================================
# 1 normal input
# =============================================================================
print('Test 1: normal input')
test_number = 1
x_mso = np.array([1.1, 1.1, -3, -5]) * R_M
y_mso = np.array([0, 0, 0, 0]) * R_M
z_mso = np.array([1, 0.5, 0.3, 0]) * R_M

r_hel = np.ones(len(x_mso)) * 0.38
di = np.ones(len(x_mso)) * 50
aberration = 0
imf_bx = imf_by = imf_bz = 0

b_KTH = kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                      control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                      dipole=True, neutralsheet=True, 
                                      ringcurrent=True, internal=True, 
                                      external=True)


desired_output = np.array([[-7.44864968e+01, -9.21868167e+01,  2.11568495e+01, -1.93035698e+01],
                           [ 0.00000000e+00,  0.00000000e+00, -1.04186710e-16,  2.23985737e-17],
                           [ 7.20433634e+01,  1.84237005e+02,  1.39022117e+01,  7.73595710e+01]])

if np.allclose(b_KTH, desired_output): 
     check_tests[test_number-1] = True  
     print('Test 1: successful')
else: 
    print('Test 1: failed! Output is not correct!')

# =============================================================================
# 2 wrong input (array length)
# =============================================================================
print('Test 2: wrong array length')
test_number = 2
x_mso = np.array([1.1, 1.1, -3]) * R_M
y_mso = np.array([0, 0, 0, 0]) * R_M
z_mso = np.array([1, 0.5, 0.3, 0]) * R_M

r_hel = np.ones(len(x_mso)) * 0.38
di = np.ones(len(x_mso)) * 50

b_KTH = kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                      control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                      dipole=True, neutralsheet=True, 
                                      ringcurrent=True, internal=True, 
                                      external=True)
if np.isnan(b_KTH) == True: 
     check_tests[test_number-1] = True 
     print('Test 2: successful')
else: 
    print('Test 2 failed! Output is not correct!')

# =============================================================================
# 3 wrong input (di and r hel out of range)
# =============================================================================
print('Test 3: Try to use wrong heliocentric distances or DI. ')
test_number = 3
x_mso = np.array([1.1, 1.1, -3, -5]) * R_M
y_mso = np.array([0, 0, 0, 0]) * R_M
z_mso = np.array([1, 0.5, 0.3, 0]) * R_M

r_hel = np.ones(len(x_mso)) * 20
di = np.ones(len(x_mso)) * 50

print('KTH output: ')
b_KTH = kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                      control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                      dipole=True, neutralsheet=True, 
                                      ringcurrent=True, internal=True, 
                                      external=True)

x_mso = np.array([1.1, 1.1, -3, -5]) * R_M
y_mso = np.array([0, 0, 0, 0]) * R_M
z_mso = np.array([1, 0.5, 0.3, 0]) * R_M

r_hel = np.ones(len(x_mso)) * 0.38
di = np.array([10, 20, 30, 150])
print('KTH output: ')

b_KTH = kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                      control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                      dipole=True, neutralsheet=True, 
                                      ringcurrent=True, internal=True, 
                                      external=True)
# =============================================================================
# 4 only points in planet
# =============================================================================
print('Test 4: Try to use only points inside Mercury. ')
test_number = 4
x_mso = np.array([1.1, 1.1, -3, -5]) 
y_mso = np.array([0, 0, 0, 0]) 
z_mso = np.array([1, 0.5, 0.3, 0]) 

r_hel = np.ones(len(x_mso)) * 0.38
di = np.ones(len(x_mso)) * 50

print('KTH output: ')

b_KTH = kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                      control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                      dipole=True, neutralsheet=True, 
                                      ringcurrent=True, internal=True, 
                                      external=True)

if b_KTH.all() == np.nan: 
    print(print('Test 4: successful'))
# =============================================================================
# 5 only points outside magnetosphere
# =============================================================================
print('Test 5: Try to use only points outside of the magnetosphere. ')
test_number = 5

x_mso = np.array([ 3, 5, 5]) * R_M
y_mso = np.array([0, 0, 0]) * R_M
z_mso = np.array([3, 5, 0]) * R_M

r_hel = np.ones(len(x_mso)) * 0.38
di = np.ones(len(x_mso)) * 50

b_KTH = kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                      control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                      dipole=True, neutralsheet=True, 
                                      ringcurrent=True, internal=True, 
                                      external=True)
# =============================================================================
# 6 fieldlinetracing
# =============================================================================
print('Test 6: Calculate fieldline. ')
test_number = 6


x_start = np.array([1.3])*R_M
y_start = np.array([0])
z_start = np.array([0]) * R_M
#r_hel = 0.39
r_hel = np.array([0.39])
di = np.array([39])

b_KTH_fieldline = kth.trace_fieldline_v10(x_start, y_start, z_start, r_hel, di, aberration, control_param_path, fit_param_path)
#print(b_KTH_fieldline)



# =============================================================================
# 1 normal input
# =============================================================================
print('Test 7: single input')
test_number = 7
x_mso = np.array([1.1]) * R_M
y_mso = np.array([0]) * R_M
z_mso = np.array([1]) * R_M

r_hel = np.ones(len(x_mso)) * 0.38
di = np.ones(len(x_mso)) * 50

b_KTH = kth.kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, 
                                      control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, 
                                      dipole=True, neutralsheet=True, 
                                      ringcurrent=True, internal=True, 
                                      external=True)
b_KTH_fieldline = kth.trace_fieldline_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, control_param_path, fit_param_path)
# #print(b_KTH_fieldline)



'''
If KTH Version 10b works correctly on your computer it should return: 
    
    
Test 1: normal input
Test 1: successful
Test 2: wrong array length
Number of positions (x,y,z) do not match
Test 2: successful
Test 3: Try to use wrong heliocentric distances or DI. 
KTH output: 
Please use r_hel (heliocentric distance) in AU, not in km. r_hel should be between 0.3 and 0.47. 
KTH output: 
At least one element in DI is greater than 100. DI must be between 0 and 100. If you don't know the exact value, use 50.
Test 4: Try to use only points inside Mercury. 
KTH output: 
Warning: 4 point(s) are located inside the planet.
Test 5: Try to use only points outside of the magnetosphere. 
No points within the magnetopause! Setting result to NaN...
Test 6: Calculate fieldline. 
i =  0
i =  10
Test 7: single input
i =  0
i =  10

'''




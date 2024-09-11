# -*- coding: utf-8 -*-

# KTH22 Model for Mercury Version 10


###################################################################################################################

# Description:
#      Calculates the magnetospheric field for Mercury. Based on Korth et al., (2015) with  improvements.
#      Model is intended for planning purposes of the BepiColombo mission. Keep the model within the team. 
#      If you plan to make a publication with the aid of this model, the opportunity to participate as co-author
#      would be appreciated. 
#      If you have suggestions for improvements, do not hesitate to write me an email.
#      
#      Takes into account:
#        - internal dipole field (offset dipole)
#        - field from neutral sheet current
#        - respective shielding fields from magnetopause currents
#        - aberration effect due to orbital motion of Mercury
#        - scaling with heliocentric distance
#        - scaling with Disturbance Indec (DI)
#      If no keywords are set, the total field from all modules will be calculated.
#
#       Required python packages: numpy, scipy
#
# Parameters:
#      x_mso: in, required, X-positions (array) in MSO base given in km
#      y_mso: in, required, Y-positions (array) in MSO base given in km
#      z_mso: in, required, z-positions (array) in MSO base given in km
#      r_hel: in, required, heliocentric distance in AU, use values between 0.3 and 0.47  
#      DI: in, required, disturbance index (0 < DI < 100), if not known:  50 (mean value) 
#      aberration: in, required,  aberration due to Mercury's orbit, use value in degrees, average: 8 degrees
#      modules: dipole (internal and external), neutralsheet (internal and external)
#      "external = True" calculates the cf-fields (shielding fields) for each module which is set true
# 
# Return: 
#     Bx, By, Bz in nT for each coordinate given (x_mso, y_mso, z_mso)
#     (if outside of MP or inside Mercury, the KTH-model will return 'nan')
#     
#      
#    :Author:
#      Daniel Heyner, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, d.heyner@tu-bs.de
#      Kristin Pump, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, k.pump@tu-bs.de
#

#   publication: https://doi.org/10.1029/2023JA031529 "Revised Magnetospheric Model Reveals Signatures of
#   Field-Aligned Current Systems at Mercury" (publication is on version 8, this is version 10)
#
#   latest update: 18th December 2023, aberration and IMF penetration 
#
###################################################################################################################

import numpy as np
import json
import scipy.special as special
from scipy.integrate import simps
import mpmath
import sys 
import matplotlib.pyplot as plt
from numba import jit, njit, prange



def kth_model_for_mercury_v10(x_mso, y_mso, z_mso, r_hel, di, aberration, control_param_path, fit_param_path, 
                               imf_bx, imf_by, imf_bz, dipole=True, neutralsheet=True, ringcurrent = True, internal=True,
                               external=True):
    print('calculating KTH Magnetic Field Version 10b')

    
    input_length = x_mso.size
    x_mso = np.array(x_mso).flatten()
    y_mso = np.array(y_mso).flatten()
    z_mso = np.array(z_mso).flatten()
    
    
    if type(x_mso) != float :
        shape = np.array(x_mso.shape) 
        if shape.size >= 2 : 
            print('Positions must be float or 1D arrays! Aborting... ')
            #sys.exit()
            return np.nan
            

    if x_mso.size != y_mso.size:
        print('Number of positions (x,y,z) do not match')
        #sys.exit()
        return np.nan
    
    if x_mso.size != z_mso.size:
        print('Number of positions (x,y,z) do not match')
        #sys.exit()
        return np.nan
        
    if x_mso.size != r_hel.size:
        print('Length of heliocentric distance (r_hel) does not match. All input array must have the same size.')
        #sys.exit()
        return np.nan
        
    if x_mso.size != di.size:
        print('Length of disturbance index (di) does not match. All input array must have the same size.')
        #sys.exit()
        return np.nan

    if internal == False and external == False:
        print('Internal and external field are both set \"False\". Set at least one True.')
        #sys.exit()
        return np.nan

    if (dipole == False and neutralsheet == False and ringcurrent == False):
        print('Dipole, neutralsheet and ringcurrent are set \"False\". Set at least one True.')
        #sys.exit()
        return np.nan
        
    print_feedback = True #if True: print output with number of positions which are inside the planet or outside the magnetosphere
                          #if False: No print output

    ############################################################################################
    #        Reading control parameters and shielding coefficients from file                   #
    ############################################################################################


    with open(control_param_path, "r") as file:
        control_params = json.load(file)


    shielding_input = fit_param_path
    shielding_par_file = open(shielding_input, "r")
    shielding_params = np.loadtxt(shielding_par_file)
    shielding_par_file.close()

    # defining the lengths of the following arrays
    n_lin_int = shielding_params[0].astype(int)   #16
    n_non_lin_int = shielding_params[1].astype(int)   #4
    n_lin_neutralcurrent = shielding_params[2].astype(int)   #0
    n_non_lin_neutralcurrent = shielding_params[3].astype(int)   #0


    # length check
    length_check = 4 + n_lin_int + 3 * n_non_lin_int + n_lin_neutralcurrent + 3 * n_non_lin_neutralcurrent

    if len(shielding_params) != length_check:
        print('Wrong shielding coefficients file length. Length has to be ' + str(length_check) + '. Length is ' + str(len(shielding_params)) + ' at the moment.' )
        #sys.exit()
        return np.nan

    # define coefficient arrays

    low = 4
    high = low + n_lin_int
    lin_coeff_int = shielding_params[low:high]
    control_params['lin_coeff_int'] = lin_coeff_int

    low = high
    high = low + n_lin_neutralcurrent
    lin_coeff_disk = shielding_params[low:high]
    control_params['lin_coeff_disk'] = lin_coeff_disk

    low = high
    high = low + n_non_lin_int
    p_i_int = shielding_params[low:high]
    control_params['p_i_int'] = p_i_int 

    low = high
    high = low + n_non_lin_neutralcurrent
    p_i_disk = shielding_params[low:high]
    control_params['p_i_disk'] = p_i_disk

    low = high
    high = low + n_non_lin_int
    q_i_int = shielding_params[low:high]
    control_params['q_i_int'] = q_i_int 

    low = high
    high = low + n_non_lin_neutralcurrent
    q_i_disk = shielding_params[low:high]
    control_params['q_i_disk'] = q_i_disk

    low = high
    high = low + n_non_lin_int
    x_shift_int = shielding_params[low:high]
    control_params['x_shift_int'] = x_shift_int 

    low = high
    high = low + n_non_lin_neutralcurrent
    x_shift_disk = shielding_params[low:high]
    control_params['x_shift_disk'] = x_shift_disk


    #######################################################################################
    # DI-Scaling
    #######################################################################################
    if len(np.atleast_1d(di)) > 1:
        if any(t > 100 for t in di):
            print('At least one element in DI is greater than 100. DI must be between 0 and 100. If you don\'t know the '
                'exact value, use 50.')
            #sys.exit()
            return np.nan

        if any(t < 0 for t in di):
            print(
             'At least one element in DI is negative. DI must be between 0 and 100. If you don\'t know the exact value, use 50.')
            #sys.exit()
            return np.nan
    elif len(np.atleast_1d(di)) == 1:
        if di < 0:
            print('Disturbance index di must be between 0 and 100. If you don\'t know the exact value, use 50.')
            #sys.exit()
            return np.nan
        if di > 100:
            print('Disturbance index di must be between 0 and 100. If you don\'t know the exact value, use 50.')
            #sys.exit()
            return np.nan


    f = control_params['f_a'] + (control_params['f_b'] * di)  # f is a factor for the scaling for R_SS (subsolar standoff distance)

    #######################################################################################
    # RMP-Scaling
    #######################################################################################
    

    if len(np.atleast_1d(r_hel)) > 1:        
        if any(r_hel > 0.47):
            print('Please use r_hel (heliocentric distance) in AU, not in km. r_hel should be between 0.3 and 0.47. ')
            #sys.exit()
            return np.nan
        if any(r_hel < 0.3):
            print('Please use r_hel (heliocentric distance) in AU, not in km. r_hel should be between 0.3 and 0.47. ')
            #sys.exit()
            return np.nan
    if len(np.atleast_1d(r_hel)) == 1:
        if r_hel > 0.47:
            print('Please use r_hel (heliocentric distance) in AU, not in km.r_hel should be between 0.3 and 0.47.')
            #sys.exit()
            return np.nan
        if r_hel < 0.3:
            print('Please use r_hel (heliocentric distance) in AU, not in km.r_hel should be between 0.3 and 0.47.')
            #sys.exit()
            return np.nan

    R_SS = f * (r_hel ** (1 / 3)) * control_params['RPL']

    #print('Achtung, line 244, R_SS2 eingeführt')
    #f2 = 2.14 + (-0.00368 * di)
    #R_SS = f2 * (r_hel ** (1 / 3)) * control_params['RPL']
    
    control_params['kappa'] = control_params['RMP'] / R_SS
    control_params['kappa3'] = (control_params['kappa']) ** 3
    

    ################################################################################################################
    # Application of the offset: MSO->MSM coordinate system
    # Scaling to planetary radius
    # Scaling with heliocentric distance
    ################################################################################################################

    dipole_offset = 479 / control_params['RPL']  # value of offset by Anderson et al. (2012)
    x_msm_in = x_mso / control_params['RPL']
    y_msm_in = y_mso / control_params['RPL']
    z_msm_in = z_mso / control_params['RPL'] - dipole_offset
    aberration  = aberration /(360) * 2* np.pi   #from degree to rad, aberration angle is negative
    
    # application of the aberration to the coordinates
    x_msm = x_msm_in * np.cos(aberration) - y_msm_in * np.sin(aberration)
    y_msm = x_msm_in * np.sin(aberration) + y_msm_in * np.cos(aberration)
    z_msm = z_msm_in                                                                                                      

    ################################################################################################################
    # Check for points lying outside the magnetopause. The magnetic field for these will be set to zero.
    # Also give a warning for calculation within the planet
    ################################################################################################################

    r_mso = np.sqrt((x_mso) ** 2 + (y_mso) ** 2 + (z_mso) ** 2) 
    r_msm = np.sqrt((x_msm) ** 2 + (y_msm) ** 2 + (z_msm) ** 2) 
    
    indices_inside_planet = np.where(r_mso < 0.99*control_params['RPL'])
    
    if indices_inside_planet[0].size > 0:  
        if print_feedback == True:
            print('Warning: ' +  str(indices_inside_planet[0].size) + ' point(s) are located inside the planet.')
      
    r_mp_check = shue_mp_calc_r_mp(x_msm, y_msm, z_msm, R_SS, control_params['alpha']) / control_params['RPL']
    


    usable_indices = np.where(r_msm <= r_mp_check)

    np.savetxt('usable_indices.txt', usable_indices)

    n_points = x_mso.size
    


    if usable_indices[0].size == 0:
        if print_feedback == True:
            print('No points within the magnetopause! Setting result to NaN...')
        
        return np.zeros((3, input_length)) * np.nan

    
    elif len([usable_indices[0]]) < n_points:
        #restrict to points within magnetopause
        x_msm = x_msm[usable_indices]
        y_msm = y_msm[usable_indices]
        z_msm = z_msm[usable_indices]
        
        control_params['kappa'] = control_params['kappa'][usable_indices]
        control_params['kappa3'] = control_params['kappa3'][usable_indices]
        di = di[usable_indices]
        
        if isinstance(aberration, np.ndarray):
            aberration = aberration[usable_indices]
        
        points_outside = n_points - usable_indices[0].size
        
        if points_outside > 0: 
            if print_feedback == True:
                print('Warning: ' +  str(n_points - usable_indices[0].size) + ' point(s) are located outside the magnetopause.')
       
    x_msm=x_msm.flatten()
    y_msm=y_msm.flatten()
    z_msm=z_msm.flatten()
    
    ##############################################################
    # Calculation of the model field
    #############################################################
    result = model_field_v10(x_msm, y_msm, z_msm, di, aberration, dipole, neutralsheet, ringcurrent, internal, external, control_params, R_SS)
    
    #set all elements to nan which are located outside the magnetosphere or inside the planet 
    result_with_nan = np.empty((3, input_length)) * np.nan  
    
    result_with_nan[0,usable_indices] = result[0]
    result_with_nan[1,usable_indices] = result[1]
    result_with_nan[2,usable_indices] = result[2]
    
    result_with_nan[0,indices_inside_planet] = np.empty(indices_inside_planet[0].size) * np.nan
    result_with_nan[1,indices_inside_planet] = np.empty(indices_inside_planet[0].size) * np.nan
    result_with_nan[2,indices_inside_planet] = np.empty(indices_inside_planet[0].size) * np.nan
    
    
    return result_with_nan

def model_field_v10(x_msm, y_msm, z_msm, di, aberration, dipole, neutralsheet, ringcurrent, internal, external, control_params, R_SS):
    # This function calls the different modules (Dipole, Ring Current, 
    # Neutral Sheet Current) and their shielding fields 
    # (external fields ). These return the magnetic field of the corresponding
    # 
    g10_int_ind = control_params['g10_int_ind']
    kappa3      = control_params['kappa3']
    g10_int     = control_params['g10_int']
    aberration  = aberration
    
    

    t_nsc = control_params['t_a'] + control_params['t_b'] * di
    #t_rc = control_params['t_a'] - control_params['t_b'] * di
    t_rc = 0.51

    n_points = x_msm.size

    # multiply with kappa
    x_msm_k = x_msm * control_params['kappa']
    y_msm_k = y_msm * control_params['kappa']
    z_msm_k = z_msm * control_params['kappa']



    if n_points == 1:
        B_total = np.zeros([3, 1])
    else : 
        B_total = np.zeros([3, n_points])
        
    ##################################################
    # calculate fields
    #################################################


    if dipole:
        if internal:

            B_int = kappa3 * internal_field_v10(x_msm_k, y_msm_k, z_msm_k, control_params)
            B_total = B_total + B_int            
            
        if external:   
             
            # This was intended to include any induced internal dipole fields. 
            # The coefficient was based on empirical work. 
            
            induction_scale_fac = -0.0052631579 * (g10_int + g10_int_ind)
            B_cf_int = kappa3 * induction_scale_fac * cf_field_v10(x_msm_k, y_msm_k, z_msm_k,
                                                                  control_params['lin_coeff_int'],
                                                                  control_params['p_i_int'], control_params['q_i_int'],
                                                                  control_params['x_shift_int'])
            #add
            B_total = B_total + B_cf_int
            

    if neutralsheet:
        if internal:
            z_offset = 0.   
                
            B_nsc = tail_field_ns_bs_v10(x_msm, y_msm, z_msm, z_offset)      
            B_total = B_total + t_nsc * B_nsc           

        if external:
            
            '''
            #add image sheets at twice the asymptotic tail radius with the opposite current direction
            
            R_tail_asymptotic = R_SS * 2. / control_params['RPL']
            
            z_offset = 2. * R_tail_asymptotic
            B_image1_ns = -tail_field_ns_bs_v10(x_msm, y_msm, z_msm, z_offset)

            z_offset = -2. * R_tail_asymptotic
            B_image2_ns = -tail_field_ns_bs_v10(x_msm, y_msm, z_msm, z_offset)
            '''
            
            #calculate the Chapman-Ferraro currents (shielding)
          
            B_cf_ns = 0.05 * cf_field_v10(x_msm_k, y_msm_k, z_msm_k, control_params['lin_coeff_disk'],
                                    control_params['p_i_disk'], control_params['q_i_disk'],
                                    control_params['x_shift_disk'])
            
            #add 
            B_total = B_total + (t_nsc * B_cf_ns)# + B_image1_ns + B_image2_ns 
            
    if ringcurrent:
        if internal:            
            
            scale_fac = 5            
            B_rc = scale_fac * tail_field_ringcurrent_v10(x_msm_k, y_msm_k, z_msm_k, di, control_params)
            B_total = B_total + t_rc *  B_rc

        if external:
            # the ring current field is so small, so that a shielding is not necessarily required. This 
            # is why the cf_disk part is deactivated (at the moment).             

            B_total = B_total 


    ##################################################
    # rotate magnetic field back to MSO base
    #################################################
    
    if x_msm.size == 1:
        bx = B_total[0]
        by = B_total[1]
        bz = B_total[2] 
    else:
        bx = B_total[0, :]
        by = B_total[1, :]
        bz = B_total[2, :]

    b_x = bx * np.cos(aberration) - by * np.sin(aberration)
    b_y = bx * np.sin(aberration) + by * np.cos(aberration)
    b_z = bz

    return np.array([b_x, b_y, b_z])

def cf_field_v10(x_msm: np.ndarray, y_msm: np.ndarray, z_msm: np.ndarray, lin_coeff: list, p_i, q_i, x_shift):
    # this function calculates the chapman-ferraro-field (schielding field) for the KTH Model
    
    n_vec = x_msm.size
 
    N = len(p_i)    

    b_x_cf = np.zeros(n_vec)
    b_y_cf = np.zeros(n_vec)
    b_z_cf = np.zeros(n_vec)

    for i_vec in range(n_vec):
        for i in range(N):
            
            for k in range(N):
                
                pq = np.sqrt(p_i[i] * p_i[i] + q_i[k] * q_i[k])
                
                lin_index = i * N + k
                

                b_x_cf[i_vec] = b_x_cf[i_vec] - pq * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])) * np.cos(p_i[i] * y_msm[i_vec]) * np.sin(
                    q_i[k] * z_msm[i_vec])
                        
                b_y_cf[i_vec] = b_y_cf[i_vec] + p_i[i] * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])) * np.sin(p_i[i] * y_msm[i_vec]) * np.sin(
                    q_i[k] * z_msm[i_vec])
                        
                b_z_cf[i_vec] = b_z_cf[i_vec] - q_i[k] * lin_coeff[lin_index] * np.exp(
                    pq * (x_msm[i_vec] - x_shift[i])) * np.cos(p_i[i] * y_msm[i_vec]) * np.cos(
                    q_i[k] * z_msm[i_vec])



    return np.array([b_x_cf, b_y_cf, b_z_cf])


def internal_field_v10(x_msm, y_msm, z_msm, control_params):

    # this calculates the magnetic field of an internal axisymmetric
    # dipole in a standard spherical harmonic expansion. The field
    # is then rotated back to the cartesian coordinate system base.

    # INPUT COORDINATES ARE IN PLANETARY RADII



    g10_int_ind = control_params['g10_int_ind']
    g10 = control_params['g10_int']

    # transform to MSO coordinates

    x_mso = np.array(x_msm)
    y_mso = np.array(y_msm)
    z_mso = np.array(z_msm) + 0.196
    #z_mso = np.array(z_msm)

    r_mso       = np.sqrt(x_mso ** 2 + y_mso ** 2 + z_mso ** 2)
    phi_mso     = np.arctan2(y_mso, x_mso)
    theta_mso   = np.arccos(z_mso / r_mso)

    # spherical harmonic synthesis of axisymmetric components
    # Daniel: higher degree coefficients from Anderson et al. 2012

    g20 = -74.6
    g30 = -22.0
    g40 = -5.7

    # l=1
    b_r_dip = 2. * (1. / r_mso) ** 3. * (g10 + g10_int_ind) * np.cos(theta_mso)
    b_t_dip = (1. / r_mso) ** 3. * (g10 + g10_int_ind) * np.sin(theta_mso)


    # l=2
    b_r_quad = 3. * (1. / r_mso) ** 4. * g20 * 0.5 * (3. * np.cos(theta_mso) ** 2. - 1.)
    b_t_quad = (1. / r_mso) ** 4. * g20 * 3. * (np.cos(theta_mso) * np.sin(theta_mso))


    # l=3
    b_r_oct = 4. * (1. / r_mso) ** 5. * g30 * 0.5 * (5. * np.cos(theta_mso) ** 3. - 3. * np.cos(theta_mso))
    b_t_oct = (1. / r_mso) ** 5. * g30 * 0.375 * (np.sin(theta_mso) + 5. * np.sin(3. * theta_mso))


    # l=4
    b_r_hex = 5. * (1. / r_mso) ** 6. * g40 * (0.125 * (35. * np.cos(theta_mso) ** 4. - 30. * np.cos(theta_mso) ** 2. + 3.))
    b_t_hex = (1. / r_mso) ** 6. * g40 * (0.3125 * (2. * np.sin(2. * theta_mso) + 7. * np.sin(4. * theta_mso)))


    # add multipoles together
    b_r = b_r_dip + b_r_quad + b_r_oct + b_r_hex
    b_t = b_t_dip + b_t_quad + b_t_oct + b_t_hex


    # rotate to mso coordinate base
    b_x_mso_int = b_r * np.sin(theta_mso) * np.cos(phi_mso) + b_t * np.cos(theta_mso) * np.cos(phi_mso)
    b_y_mso_int = b_r * np.sin(theta_mso) * np.sin(phi_mso) + b_t * np.cos(theta_mso) * np.sin(phi_mso)
    b_z_mso_int = b_r * np.cos(theta_mso) - b_t * np.sin(theta_mso)


    return np.array([b_x_mso_int, b_y_mso_int, b_z_mso_int])

        
def a_phi_hankel_v10(H_current, rho_z_in, phi, z, lambda_arr, d_0):
    # This function calculates the vector potential a_phi with the results from the Hankel transformation of
    # the neutral sheet current.

    sheet_thickness = d_0 
    
    integrand = H_current * special.j1(lambda_arr * rho_z_in) * np.exp( -lambda_arr * np.sqrt(z ** 2 + sheet_thickness ** 2))
  
    result_a_phi_hankel = simps(integrand, x=lambda_arr)
    
    return result_a_phi_hankel


# old:     
# def fx(xj):
#     # xj is considered in units of planetary radii
#     #result is in nA/m^2
#     x_1D = np.reshape(xj, xj.size ) 
#     result_1D = np.zeros(x_1D.size)
    
#     good_indices = np.array(np.where(x_1D < -1.))
#     if good_indices.size >= 1: 
#         result_1D[good_indices] = 100 * (x_1D[good_indices] + 1.)**2 * np.exp(-0.39*(x_1D[good_indices] + 1.)**2)
    
#     return np.reshape(result_1D, xj.shape)


def fx(xj):
    # xj is considered in units of planetary radii
    #result is in nA/m^2
    #new in Version 10: NSC is stretched out on the night side, parameters: TBD in more detail, 
    # problem: no messenger Data to compare with
    
    plateau_length = 3 #R_M 
    max_curr_loc = -2.59
    plateau_hight = 100 * (max_curr_loc + 1.)**2 * np.exp(-0.39*(max_curr_loc + 1.)**2)
    
    x_1D = np.reshape(xj, xj.size ) 

    result_1D = 100 * (x_1D + 1.)**2 * np.exp(-0.39*(x_1D + 1.)**2)
    result_1D[np.where(x_1D > -1)] = 0   # no current in planet or on dayside
   
    plateau_end = -2.59- plateau_length
    
    indices_plateau = np.where((x_1D < max_curr_loc)&(x_1D > plateau_end))    
    indices_far_tail = np.where(x_1D < plateau_end)
    
    result_1D[indices_plateau] = plateau_hight    
    result_1D[indices_far_tail] =  100 * (x_1D[indices_far_tail] +plateau_length + 1.)**2 * np.exp(-0.39*(x_1D[indices_far_tail] + plateau_length + 1.)**2)

    return np.reshape(result_1D, xj.shape)
    

def tail_field_ns_bs_v10(x_target,y_target,z_target, z_offset):
    
    mu = 4e-7 * np.pi 
    RPL = 2440e3
    
    #bounds for the integration over j_y
    x_bounds = [-5.*RPL, -1.*RPL]
    y_bounds = [-1.*RPL, 1.*RPL]
    z_bounds = [-1.*RPL + z_offset*RPL, 1.*RPL+ z_offset*RPL]

    # number of grid points in each dimension
    x_steps = 80
    y_steps = 40
    z_steps = 40   

    
    #differential volumen
    dV = (x_bounds[1] - x_bounds[0]) / float(x_steps-1.) * (y_bounds[1] - y_bounds[0]) / float(y_steps-1.) * (z_bounds[1] - z_bounds[0]) / float(z_steps-1.)
                   

    #create integration mesh with meshgrid 
    x_coords_1d = np.arange(x_steps) / float(x_steps-1) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    y_coords_1d = np.arange(y_steps) / float(y_steps-1) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
    z_coords_1d = np.arange(z_steps) / float(z_steps-1) * (z_bounds[1] - z_bounds[0]) + z_bounds[0]
    
    #prepare coordinates in multidimensional arrays - this information goes into the f_xyz functions that define the current density
    x_coords, y_coords, z_coords = np.meshgrid(x_coords_1d, y_coords_1d, z_coords_1d)
    
    x_coords = np.reshape(x_coords, x_coords.size)
    y_coords = np.reshape(y_coords, y_coords.size)
    z_coords = np.reshape(z_coords, z_coords.size)
    
    n_vec = len(x_target)    
    
    fx_result = fx(x_coords/RPL)
    
    # @njit(fastmath=True, parallel=True) 
    # def tail_field_fast_sum_multi():   
    #     Bx_arr = np.zeros(n_vec)
    #     By_arr = np.zeros(n_vec)
    #     Bz_arr = np.zeros(n_vec)
             
        
    #     #differential volume
          
    #     for i_vec in prange(n_vec):                                                                                                                                                                      
            
    #         x_rel = x_target[i_vec]*RPL - x_coords 
    #         y_rel = y_target[i_vec]*RPL - y_coords
    #         z_rel = z_target[i_vec]*RPL - z_coords
            
    #         r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)                                                                                                              
            
    #         #check for singularity
    #         for i in range(len(r)) :
    #             if r[i] < (0.1*RPL): 
    #                 r[i] = 0.1*RPL                                                                                                                                                                                                                                 
    #         r_inv = r**(-3.)
                                                                                                        
    #         Bx = mu / (4. * np.pi) * np.sum( fx_result * np.exp(-0.5*(y_coords/RPL)**2) * np.exp(-0.1 * ((z_coords/RPL) / 0.12)**2) * z_rel * r_inv) * dV 
    #         Bz = mu / (4. * np.pi) * np.sum(-fx_result * np.exp(-0.5*(y_coords/RPL)**2) * np.exp(-0.1 * ((z_coords/RPL) / 0.12)**2) * x_rel * r_inv) * dV
    #         Bx_arr[i_vec] = Bx 
    #         Bz_arr[i_vec] = Bz
            
    #     return Bx_arr, By_arr, Bz_arr 

        
    @njit(fastmath=True, parallel=True)
    def tail_field_fast_sum_multi():
        # Pre-allocate arrays
        B_arr = np.zeros((3, n_vec))         
        
        # differential volume
          
        for i_vec in prange(n_vec):
            x_rel = x_target[i_vec] * RPL - x_coords
            y_rel = y_target[i_vec] * RPL - y_coords
            z_rel = z_target[i_vec] * RPL - z_coords
            
            r = np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)

            # Check for singularity
            for i in range(len(r)):
                if r[i] < (0.1 * RPL):
                    r[i] = 0.1 * RPL
            
            r_inv = r ** (-3.0)
                                    
 
            # Calculate Bx and Bz (By remains zero)
            common_factor = np.exp(-0.5 * (y_coords / RPL) ** 2) * np.exp(-0.1 * ((z_coords / RPL) / 0.12) ** 2)
            Bx = mu / (4.0 * np.pi) * np.sum(fx_result * common_factor * z_rel * r_inv) * dV
            Bz = mu / (4.0 * np.pi) * np.sum(-fx_result * common_factor * x_rel * r_inv) * dV
                                                                                                                                                       
            B_arr[0, i_vec] = Bx  # Bx
            B_arr[2, i_vec] = Bz  # Bz
        
        return B_arr        
       
    def tail_field_fast_sum_single():   
        Bx_arr = np.zeros(n_vec)
        By_arr = np.zeros(n_vec)
        Bz_arr = np.zeros(n_vec)
             
        
        #differential volume
          
        for i_vec in range(n_vec):            
            
            x_rel = x_target[i_vec]*RPL - x_coords 
            y_rel = y_target[i_vec]*RPL - y_coords
            z_rel = z_target[i_vec]*RPL - z_coords
            
            r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2) 
            
            #check for singularity
            for i in range(len(r)) :
                if r[i] < (0.1*RPL): 
                    r[i] = 0.1*RPL                                 
            
            r_inv = r**(-3.)
            
            Bx = mu / (4. * np.pi) * np.sum( fx_result * np.exp(-0.5*(y_coords/RPL)**2) * np.exp(-0.1 * ((z_coords/RPL) / 0.12)**2) * z_rel * r_inv) * dV 
            Bz = mu / (4. * np.pi) * np.sum(-fx_result * np.exp(-0.5*(y_coords/RPL)**2) * np.exp(-0.1 * ((z_coords/RPL) / 0.12)**2) * x_rel * r_inv) * dV
            Bx_arr[i_vec] = Bx 
            Bz_arr[i_vec] = Bz
            
        return Bx_arr, By_arr, Bz_arr
        
       
    # decision wether single or multi processing is used to make the model faster.
    # On test computer ( Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz ), multi processing is faster 
    # when more than 70 data points are calculated. This value might differ on other
    # computers. 
    
    single_multi_limit = 70
    
    if x_target.size > single_multi_limit: 
        #print('multi')
        output_tail_field =  tail_field_fast_sum_multi()
        
    else: 
        #print('single')
        output_tail_field =  tail_field_fast_sum_single()
        
    # Convert B_nsc to a numpy array if it's a list or tuple
    if isinstance(output_tail_field, (list, tuple)):
        output_tail_field = np.array(output_tail_field)                                                        
    return output_tail_field 


def tail_field_ringcurrent_v10(x_msm, y_msm, z_msm, di, control_params):
    #this function calculates the magnetic field of the eastward flowing ring shaped current close to the planet 

    rho = np.sqrt(x_msm ** 2 + y_msm ** 2)
    phi = np.arctan2(y_msm, x_msm)

    d_0 = control_params['d_0']

    mu_0 = 1.0
    steps = 100
    rho_min = 0.5 #these values are adapted for the specific current profile for the ring current
    rho_max = 2
    h_steps = 100  # This value is from experience. When you change the current profile this should be checked again for sufficient convergence.

    rho_hankel = np.arange(steps) / float(steps-1.) * (rho_max - rho_min) + rho_min

    current = current_profile_ringcurrent_v10(rho_hankel, control_params)

    
    lambda_max = 20  # std value
    lambda_min = 10 ** (-1)  # std value

    lambda_out = 10 ** (np.divide(range(h_steps), (float(h_steps) - 1)) * (
            np.log10(lambda_max) - np.log10(lambda_min)) + np.log10(lambda_min))
        

    result_hankel_trafo = np.zeros(h_steps)
    for i in range(h_steps):
        #special.j1 = Bessel function of the first kind of order 1
        result_hankel_trafo[i] = simps(special.j1(lambda_out[i] * rho_hankel) * current * rho_hankel,
                                       x=rho_hankel, axis = -1)  
        
    H_current = mu_0 / 2.0 * result_hankel_trafo
    

    ###############################################################

    n_vec = len(np.atleast_1d(x_msm))
    b_rc_x = np.zeros(n_vec)
    b_rc_y = np.zeros(n_vec)
    b_rc_z = np.zeros(n_vec)
    b_rc_rho = np.zeros(n_vec)

  
    for i in range(n_vec):
        a_phi = a_phi_hankel_v10(H_current, rho[i], phi[i], z_msm[i], lambda_out, d_0)

        # numerically approximate the derivatives
        delta_z = 10 ** (-5)

        d_a_phi_d_z = (a_phi_hankel_v10(H_current, rho[i], phi[i], z_msm[i] + delta_z, lambda_out, d_0) - a_phi_hankel_v10(H_current, rho[i],
                                                                    phi[i],
                                                                    z_msm[i] - delta_z,
                                                                    lambda_out,
                                                                    d_0)) / (
                          2 * delta_z)

        delta_rho = 10 ** (-5)
        d_a_phi_d_rho = (a_phi_hankel_v10(H_current, rho[i] + delta_rho, phi[i], z_msm[i], lambda_out,
                                        d_0) - a_phi_hankel_v10(
            H_current, rho[i] - delta_rho, phi[i],
            z_msm[i], lambda_out, d_0)) / (2 * delta_rho)


        b_rc_rho[i] =  (- d_a_phi_d_z)
        
        
        

        if rho[i] <= 10 ** (-4):
            b_rc_z[i] =  (1.0 + d_a_phi_d_rho)

        else:
            b_rc_z[i] =  (a_phi / rho[i] + d_a_phi_d_rho)

        # rotate back to cartesian
        b_rc_x[i] = b_rc_rho[i] * np.cos(phi[i])
        b_rc_y[i] = b_rc_rho[i] * np.sin(phi[i])

    #z_sheet_thickness =   1 * np.exp(-0.5*((z_msm)/2)**2 )
    # rho_sheet_thickness =  1 * np.exp(-0.5*((rho-1)/2)**2 )
    z_sheet_thickness = 1
    
    
    b_rc_z = b_rc_z * z_sheet_thickness 
    b_rc_x = b_rc_x * z_sheet_thickness 
    b_rc_y = b_rc_y * z_sheet_thickness
    


    return np.array([b_rc_x, b_rc_y, b_rc_z])

def current_profile_ringcurrent_v10(rho, control_params):
    #this function calculates the current profile of the ring current
    
    d = control_params['d']
    e = control_params['e']
    f = control_params['f']
    
    
    current = -(d/f*np.sqrt(2*np.pi)) * np.exp(-(rho-e)**2/(2*f**2))
    
    
    return current

def shue_mp_calc_r_mp(x, y, z, RMP, alpha):
	"""
	This calculates the magnetopause distance after the Shue et al. magnetopause model
	for the radial extension of an arbitrary point.
	
	x,y,z : coordinates - arbitrary units in MSM coordinate system
	RMP : subsolar standoff distance - arbitrary units --> result will have the same units
	alpha : mp flaring parameter
	
	return : magnetopause distance w.r.t. planetary center 
	"""
	#distance to x-axis
	rho_x = np.sqrt(y**2 + z**2)
	#angle with x-axis
	epsilon = np.arctan2(rho_x,x)
	
	
	#Shue's formula
	mp_distance = RMP * np.power((2. / (1. + np.cos(epsilon))),alpha)
	
	return mp_distance
	


def mp_normal_v10(x_msm, y_msm, z_msm, RMP, alpha):
    """
	This function calculates the normal vector at the magnetopause.
	
	x,y,z : coordinates - arbitrary units in MSM coordinate system
	RMP : subsolar standoff distance - arbitrary units --> result will have the same units
	alpha : mp flaring parameter
	
	return : x, y, z components of normal vector (on MP) 
	"""
    
    r       = np.sqrt(x_msm**2 + y_msm**2 + z_msm**2)
    phi     = np.arctan2(y_msm, x_msm)
    theta   = np.arccos(z_msm / r)
    
    r_mp = shue_mp_calc_r_mp(x_msm, y_msm, z_msm, RMP, alpha)
    mp_loc_x = r_mp * np.sin(theta) * np.cos(phi)
    mp_loc_y = r_mp * np.sin(theta) * np.sin(phi)
    mp_loc_z = r_mp * np.cos(theta)
    
    # first tangential vector: along rotation of gamma (rotation axis: x-axis)
    gamma     = np.arctan2(mp_loc_y, mp_loc_z)
    e_gamma_x = 0.
    e_gamma_y = np.cos(gamma)
    e_gamma_z = - np.sin(gamma)
    
    # second tangential vector: along the change of epsilon. This does NOT change gamma
    epsilon     = np.cos(2. * ((r_mp / RMP)**(- 1. / alpha)) - 1.)
    d_epsilon   = 1e-3
    new_epsilon = epsilon + d_epsilon
    
    #with the new epsilon angle, calculate the corresponding magnetpause position
    new_r_mp      = RMP * (2. / (1. + np.cos(new_epsilon)))**alpha
    new_mp_loc_x  = new_r_mp * np.cos(new_epsilon)
    new_mp_loc_y  = new_r_mp * np.sin(new_epsilon) * np.sin(gamma)
    new_mp_loc_z  = new_r_mp * np.sin(new_epsilon) * np.cos(gamma)
    
    #the difference vector is the connection vector
    connect_x = new_mp_loc_x - mp_loc_x
    connect_y = new_mp_loc_y - mp_loc_y
    connect_z = new_mp_loc_z - mp_loc_z
    
    #normalize and take the opposite direction
    magnitude = np.sqrt(connect_x**2 + connect_y**2 + connect_z**2)
  
    connect_x = -connect_x / magnitude
    connect_y = -connect_y / magnitude
    connect_z = -connect_z / magnitude
    
    #get normal direction by cross-product of tangentials
    #since both vectors are nomalized to 1 the cross-product has also the length 1
    mp_normal_x = connect_y * e_gamma_z - e_gamma_y * connect_z
    mp_normal_y = connect_z * e_gamma_x - e_gamma_z * connect_x
    mp_normal_z = connect_x * e_gamma_y - e_gamma_x * connect_y

    return mp_normal_x, mp_normal_y, mp_normal_z

def trace_field_line_single_v10():
    mercury1 = plt.Circle((0, (479/2440)), 1, color = '0.75')

    R_M = 2440.
    x_start= -1.2*R_M
    y_start = 0
    z_start = 479
    r_hel = 0.37
    di = 50
    

    fieldlinetrace = trace_fieldline_v10(x_start, y_start, z_start, r_hel, di, delta_t=0.3)
    fieldlinetrace = trace_fieldline_v10(x_start, y_start, z_start, r_hel, di, delta_t=-0.3)
    #print(fieldlinetrace)
    x = fieldlinetrace[0]   #x
    y = fieldlinetrace[1]   #y
    z = fieldlinetrace[2]   #z
    


    fig, ax1 = plt.subplots()
    ax1.add_artist(mercury1) 
    plt.plot(x/R_M, z/R_M)
    ax1.axis('square')
    plt.xlim((-5, 3))
    plt.ylim((-3, 3)) 
    ax1.grid()
    ax1.invert_xaxis()
    
def trace_fieldline_v10(x_start, y_start, z_start, r_hel, di, control_param_path, fit_param_path, delta_t=0.3):
    #for opposite direction choose delta_t = -0.3 (or smaller/higher)

    R_M = 2440  #Radius of Mercury in km
    r = np.sqrt((x_start / R_M) ** 2 + (y_start / R_M) ** 2 + (z_start / R_M) ** 2)
    
    if delta_t > 0: 
        sign = 1
    if delta_t < 0: 
        sign = -1

    if r < 1:
        print('Radius of start point is smaller than 1. You start inside the planet! Radius = ', r)
        sys.exit()

    aberration = 0
    imf_bx = imf_by = imf_bz = 0

    def f(x, y, z):
        return kth_model_for_mercury_v10(x, y, z, r_hel, di, aberration, control_param_path, fit_param_path, imf_bx, imf_by, imf_bz, True, True, True, True, True)

    x_trace = [x_start]
    y_trace = [y_start]
    z_trace = [z_start]
    mag_B_list = np.array([])
    
    x_array = np.asarray(x_trace)
    y_array = np.asarray(y_trace)
    z_array = np.asarray(z_trace)
    mag_B_list = np.asarray(mag_B_list)

    i = 0
    

    while r > 1 and i < 1000:
        
        if i%10 == 0: 
            print('i = ', i)
            
        r = np.sqrt((x_trace[i] / R_M) ** 2 + (y_trace[i] / R_M) ** 2 + (z_trace[i] / R_M) ** 2)
        if r < 1.00:
            print('r < 1RM')
            break         
        
        B = f(x_trace[i], y_trace[i], z_trace[i])   
        
        mag_B = float(np.sqrt(B[0]**2 + B[1]**2 + B[2]**2))
        
        if r > 1.5: 
            
            delta_t = 150/mag_B            
            if delta_t < 0.3: 
                delta_t = 0.3
            elif delta_t > 8: 
                delta_t = 8
                
            delta_t = sign * delta_t
        elif r < 1.5: 
            delta_t = sign * 0.9
       
        
        if np.isnan(B[0]) == True: 
            break        
        
        if np.isnan(x_trace[i])== True: 
            break
        
        k1 = delta_t * B
        
        if np.isnan(k1[0])== True: 
            #print('k1 is nan')
            break
        
        try: 
            k2 = delta_t * f(x_trace[i] + 0.5 * k1[0], y_trace[i] + 0.5 * k1[1], z_trace[i] + 0.5 + k1[2])
        except: 
            #print('k2 is nan')
            break
    
        if np.isnan(k2[0])== True: 
            break
       
        try:
            k3 = delta_t * f(x_trace[i] + 0.5 * k2[0], y_trace[i] + 0.5 * k2[1], z_trace[i] + 0.5 * k2[2])
        except: 
            #print('k3 is nan')
            break
        if np.isnan(k3[0])== True: 
            #print('k3 is nan')
            break
    
        
        
        try: 
            k4 = delta_t * f(x_trace[i] + k3[0], y_trace[i] + k3[1], z_trace[i] + k3[2])
        except: 
            #print('k4 is nan')
            break        
                
        if np.isnan(k4[0])== True: 
            #print('k4 is nan')
            break
        
        #print('k1: ', k1)
        #print('k2: ', k2)
        #print('k3: ', k3)
        #print('k4: ', k4)
        
        
        x_trace.append(x_trace[i] + (1 / 6) * (k1[0] + 2 * k2[0] + 3 * k3[0] + k4[0]))
        y_trace.append(y_trace[i] + (1 / 6) * (k1[1] + 2 * k2[1] + 3 * k3[1] + k4[1]))
        z_trace.append(z_trace[i] + (1 / 6) * (k1[2] + 2 * k2[2] + 3 * k3[2] + k4[2]))
        mag_B_list = np.append(mag_B_list, mag_B)
        
        if np.isnan(x_trace[-1])== True: 
            #print('last element is nan')
            break

        i = i + 1

        x_array = np.asarray(x_trace)
        y_array = np.asarray(y_trace)
        z_array = np.asarray(z_trace)
        mag_B_list = np.asarray(mag_B_list)  
        
        
        
        usable_indices = np.loadtxt('usable_indices.txt')
        if len(np.atleast_1d(usable_indices)) == 0: 
            break
        
        r = np.sqrt(x_array[-1]**2 + y_array[-1]**2 + z_array[-1]**2)
        #print(r)
        
        if r < 1.00 * R_M: 
            break 
        
        if x_array[-1] <= -5 * R_M: 
            break  
        

    return (np.array([x_array, y_array, z_array]))
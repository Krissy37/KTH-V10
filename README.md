# KTH22-model

This model calculates the magnetic field inside the Hermean magnetosphere. 

Publication: Pump et al. 2024, "Revised Magnetospheric Model Reveals Signatures of
Field-Aligned Current Systems at Mercury",  https://doi.org/10.1029/2023JA031529

Description:
The model is designed to calculate the magnetospheric magnetic field for Mercury. 
Based on Korth et al., (2015) with  improvements.
Model is intended for planning purposes of the BepiColombo mission. 
If you plan to make a publication with the aid of this model, the opportunity to participate as co-author
would be appreciated. 
If you have suggestions for improvements, do not hesitate to write me an email (k.pump@tu-bs.de).
     
Takes into account:
- internal dipole field and higher degrees (offset dipole) 
- field from neutral sheet current
- eastward ring current 
- respective shielding fields from magnetopause currents
- aberration effect due to orbital motion of Mercury
- scaling with heliocentric distance
- scaling with Disturbance Indec (DI)


Required python packages: numpy, scipy

# Input Parameters:
-x_mso: in, required, X-positions (array) in MSO base given in km
-y_mso: in, required, Y-positions (array) in MSO base given in km
-z_mso: in, required, z-positions (array) in MSO base given in km
-r_hel: in, required, heliocentric distance in AU, use values between 0.3 and 0.47  
-DI: in, required, disturbance index (0 < DI < 100), if not known:  50 (mean value) 
-aberration: in, required, in degrees. if not known: 0 
-imf_bx, imf_by, imf_bz: in, not ready to use yet. Use 0. 
-modules: dipole (internal and external), neutralsheet (internal and external)
-"external = True" calculates the cf-fields (shielding fields) for each module which is set true
 
# Return: 
 Bx, By, Bz in nT for each coordinate given (x_mso, y_mso, z_mso)
(if outside of MP or inside Mercury, the KTH-model will return 'nan')

# Additional functions: 
- fieldlinetracer
- R_SS calculator
- L-Shell calculator
- M-Shell calculator
- Magnetopause distance calculator
- Aberration estimator

  All these functions have a description in kth_model_for_mercury_v10b.py.
  If you want to use them, it is recommended to look at the example files.
  There you can find more detailed descriptions. 
   

      
# Authors:
Daniel Heyner, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, d.heyner@tu-bs.de
Kristin Pump, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, k.pump@tu-bs.de

import math
from numba import cuda

'''
Amplitude sqaured

In most cases the cross sections are given instead of the S-matrix.
In these conditions, we use the differential cross sections.    

Note that the amplitude square must be strictly corresponded to the collision types.
'''
    
##########################################################################
##########################################################################
# Only modify the functions entitled sdigma_domega
# and the selection line in function Amplitude_square

'''
H+ (0), H2 (1), H (2), H2+ (3), e- (4)

for 2-2 collisions:
        (type 0): H+ + H2 <-> H  + H2+
        (type 1): H+ + H2 <-> H+ + H2
        (type 2): H2 + e- <-> e- + H2
'''

##########################################################################
##########################################################################
        

@cuda.jit(device=True)
def Amplitude_square(m1_squared,m2_squared,m3_squared,mp_squared,\
                     k10,k20,k30,p0,collision_type_indicator,\
                     k1x,k1y,k1z,k2x,k2y,k2z,k3x,k3y,k3z,px,py,pz,hbar,c,lambdax):
    
    s = m1_squared*c**2 + m2_squared*c**2 + 2*(k10*k20-k1x*k2x-k1y*k2y-k1z*k2z)
    I = math.sqrt((k10*k20-k1x*k2x-k1y*k2y-k1z*k2z)**2-m1_squared*m2_squared*c**4)
    
    ##########################################################
    ##########################################################
    # Only modify here
    if collision_type_indicator==0:
        return 10000.

    elif collision_type_indicator==1:
        return 10000.

    elif collision_type_indicator==2:
        return 10000.
    
    else:
        return 0.
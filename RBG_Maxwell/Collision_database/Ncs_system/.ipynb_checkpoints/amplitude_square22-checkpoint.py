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
Fe3+ (0), He (1)

for 2-2 collisions:
        (type 0): Fe3+ + He <-> He  + Fe3+

'''

##########################################################################
##########################################################################
        
@cuda.jit(device=True)
def a_q1q2_q1q2(d_sigma,k30,I,hbar,de):
    return d_sigma*64*math.pi**2*k30*I/hbar**2/de


@cuda.jit(device=True)
def Amplitude_square(m1_squared,m2_squared,m3_squared,mp_squared,\
                     k10,k20,k30,p0,collision_type_indicator,\
                     k1x,k1y,k1z,k2x,k2y,k2z,k3x,k3y,k3z,px,py,pz,hbar,c,lambdax):
    de = math.sqrt(px**2+py**2+pz**2)
    I = math.sqrt((k10*k20-k1x*k2x-k1y*k2y-k1z*k2z)**2-m1_squared*m2_squared*c**4)
    d_sigma = 2.53043*10**(-14)
    ##########################################################
    ##########################################################
    # Only modify here
    if collision_type_indicator==0:
        return a_q1q2_q1q2(d_sigma,k30,I,hbar,de)

    else:
        return 0.
import math
from numba import cuda

'''
Amplitude sqaured

In most cases the cross sections are given instead of the S-matrix.
In these conditions, we use the differential cross sections.    

Note that the amplitude square must be strictly corresponded to the collision types.

(type 0): a + a <-> b + b + b
'''
# differential cross section   
@cuda.jit(device=True)
def collision_type_0():
    return 10.

@cuda.jit(device=True)
def Amplitude_square(m1_squared,m2_squared,m3_squared,m4_squared,mp_squared,\
                                                    k10,k20,k30,k40,p0,collision_type_indicator,\
                                                    k1x,k1y,k1z,k2x,k2y,k2z,k3x,k3y,k3z,k4x,k4y,k4z,\
                                                    px,py,pz,hbar,c,lambdax,half_px, half_py, half_pz,flavor, i_collision):
    
    # final and initial summed amplitude squared
    # the relation between amplitude square and differential cross section is given by
    # diff_cross*64*math.pi**2*k30*I/hbar**2/math.sqrt(px**2+py**2+pz**2)
    I = math.sqrt((k10*k20-k1x*k2x-k1y*k2y-k1z*k2z)**2-m1_squared*m2_squared*c**4)
    de = math.sqrt(px**2+py**2+pz**2)
    if de < 10**(-13):
        return 0.
    else:
        if collision_type_indicator==0:
            # a + a <-> b + b + b
            diff_cross = collision_type_0()
            return diff_cross*4*(2*math.pi)**5*k30*k40*I/(half_px[flavor[i_collision,3]]*half_py[flavor[i_collision,3]]*half_pz[flavor[i_collision,3]])/de

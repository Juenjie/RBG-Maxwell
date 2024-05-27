import math
from numba import cuda

'''
Amplitude sqaured

In most cases the cross sections are given instead of the S-matrix.
In these conditions, we use the differential cross sections.    

Note that the amplitude square must be strictly corresponded to the collision types.
'''
    
# @cuda.jit(device=True)
# def a_q1q2_q1q2(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
#     return 8*(g**4)*(dF**2)*(CF**2)/dA*((s*s+u*u)/(t-mg_regulator_squared)**2)
        
# @cuda.jit(device=True)
# def a_qq_qq(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
#     return 8*(g**4)*(dF**2)*(CF**2)/dA*((s*s+u*u)/((t-mg_regulator_squared)**2)+\
#                                          (s*s+t*t)/((u-mg_regulator_squared)**2))+\
#            16*(g**4)*dF*CF*(CF-CA/2)*(s*s)/((t-mg_regulator_squared)*(u-mg_regulator_squared))
    
# @cuda.jit(device=True)
# def a_qqbar_qqbar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
#     return 8*(g**4)*(dF**2)*(CF**2)/dA*((s*s+u*u)/((t-mg_regulator_squared)**2)+\
#                                          (u*u+t*t)/(s*s))+\
#            16*(g**4)*dF*CF*(CF-CA/2)*(u*u)/((t-mg_regulator_squared)*s)
    
# @cuda.jit(device=True)
# def a_q1q1bar_q2q2bar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
#     return 8*(g**4)*(dF**2)*(CF**2)/dA*((t*t+u*u)/(s*s))
    
# @cuda.jit(device=True)
# def a_q1q1bar_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
#     return 8*g**4*dF*CF**2*(u/(t-mg_regulator_squared)+t/(u-mg_regulator_squared))-\
#            8*g**4*dF*CF*CA*(t**2+u**2)/(s**2)
    
# @cuda.jit(device=True)
# def a_qg_qg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
#     return -8*g**4*dF*CF**2*(u/s+s/(u-mg_regulator_squared))+\
#             8*g**4*dF*CF*CA*(s*s+u*u)/((t-mg_regulator_squared)**2)
    
# @cuda.jit(device=True)
# def a_gg_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
#     return 16*g**4*dA*CA**2*(3-s*u/((t-mg_regulator_squared)**2)
#                              -s*t/((u-mg_regulator_squared)**2)
#                              -t*u/(s**2))

@cuda.jit(device=True)
def Amplitude_square(m1_squared,m2_squared,m3_squared,m4_squared,mp_squared,\
                                                    k10,k20,k30,k40,p0,collision_type_indicator,\
                                                    k1x,k1y,k1z,k2x,k2y,k2z,k3x,k3y,k3z,k4x,k4y,k4z,\
                                                    px,py,pz,hbar,c,lambdax):
    return 0.
#     # amplitude square is Lorentz invariant, hence we boost the energy and momentum
#     # to the center-of-mass frame for convinience
    
#     alpha_s = 0.3
#     dF = 3.
#     CA = 3.
#     CF = 4/3
#     dA = 8.
#     g = math.sqrt(4*math.pi*alpha_s)
#     mg_regulator_squared = 0.5
    
# #     # Lorentz boost to center-of-mass frame
# #     ECM = E1+E2
# #     vx, vy, vz = (k1x+k2x)/ECM, (k1y+k2y)/ECM, (k1z+k2z)/ECM
# #     vNorm = math.sqrt(vx**2+vy**2+vz**2)
# #     vxN, vyN, vzN = vx/vNorm, vy/vNorm, vz/vNorm
# #     gamma = 1/math.sqrt(1-vNorm**2)
    
# #     E1p = (E1 - k1x*vx - k1y*vy - k1z*vz)*gamma
# #     vNDotk1 = vxN*k1x + vyN*k1y + vzN*k1z
# #     k1xp = k1x + (gamma - 1)*vNDotk1*vxN - gamma*vx*E1
# #     k1yp = k1y + (gamma - 1)*vNDotk1*vyN - gamma*vy*E1
# #     k1zp = k1z + (gamma - 1)*vNDotk1*vzN - gamma*vz*E1
    
# #     E2p = (E2 - k2x*vx - k2y*vy - k2z*vz)*gamma
# #     vNDotk2 = vxN*k2x + vyN*k2y + vzN*k2z
# #     k2xp = k2x + (gamma - 1)*vNDotk2*vxN - gamma*vx*E2
# #     k2yp = k2y + (gamma - 1)*vNDotk2*vyN - gamma*vy*E2
# #     k2zp = k2z + (gamma - 1)*vNDotk2*vzN - gamma*vz*E2
    
# #     E3p = (E3 - k3x*vx - k3y*vy - k3z*vz)*gamma
# #     vNDotk3 = vxN*k3x + vyN*k3y + vzN*k3z
# #     k3xp = k3x + (gamma - 1)*vNDotk3*vxN - gamma*vx*E3
# #     k3yp = k3y + (gamma - 1)*vNDotk3*vyN - gamma*vy*E3
# #     k3zp = k3z + (gamma - 1)*vNDotk3*vzN - gamma*vz*E3

# #     Epp = (Ep - px*vx - py*vy - pz*vz)*gamma
# #     vNDotp = vxN*px + vyN*py + vzN*pz
# #     pxp = px + (gamma - 1)*vNDotp*vxN - gamma*vx*Ep
# #     pyp = py + (gamma - 1)*vNDotp*vyN - gamma*vy*Ep
# #     pzp = pz + (gamma - 1)*vNDotp*vzN - gamma*vz*Ep
    
#     # Mandelstam variables s t and u
#     # add a regulator for t and u channels
#     # in QGP system, matrix elements are defined in terms of the
#     # energies
#     E1 = k10
#     E2 = k20
#     E3 = k30
#     Ep = p0
#     s = (E1+E2)**2-(k1x+k2x)**2-(k1y+k2y)**2-(k1z+k2z)**2
#     t = (E1-E3)**2-(k1x-k3x)**2-(k1y-k3y)**2-(k1z-k3z)**2
#     u = (E1-Ep)**2-(k1x-px)**2-(k1y-py)**2-(k1z-pz)**2

#     # spin and color averaged
#     # amplitude squared
#     if collision_type_indicator==0:
#         # q1+q2->q1+q2 
#         return a_q1q2_q1q2(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)/6**4
#     elif collision_type_indicator==1:
#         # q+q->q+q  
#         return a_qq_qq(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)/6**4
#     elif collision_type_indicator==2:
#         # q+q_bar -> q+q_bar
#         return a_qqbar_qqbar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)/6**4
#     elif collision_type_indicator==3:
#         # q1+q1_bar -> q2+q2_bar
#         return a_q1q1bar_q2q2bar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)/6**4
#     elif collision_type_indicator==4:
#         # q1+q1_bar -> gg
#         return a_q1q1bar_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)/(6**2*16**2)
#     elif collision_type_indicator==5:
#         # q+g -> q+g
#         return a_qg_qg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)/(6**2*16**2)
#     elif collision_type_indicator==6:
#         # gg -> gg
#         return a_gg_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)/16**4
from numba import float64
import math
from numba import cuda        
from numba.cuda.random import xoroshiro128p_uniform_float64
                
'''
Device function to accumulate collision term.

This function calculate ck(k1+k2-->k3+p) and save the values for all four particles,
hence enhance the number of samples by 4 times.

In this unit cell, the distribution function is given;

k3x, k3y, k3z, k1x, k1y, sampled randomly in the entire
momentum box from function collision_term_at_specific_point()
are given;

px, py, pz, sampled randomly in the small momentum grid from
function collision_term_at_specific_point() are given;
Note that this random sample is different from the sample
of k3x, k3y, k3z, k1x, k1y. The sample of p is in the
specific momentum grid, which corresponds to the scan in the left handside
of Boltzmann Equation df(p,x,t)/dt = ... While the sample of k3 and k1 is in
the entire momentum box which corresponds to the direct Monte Carlo integration.

k1z is obtained by solving E1 + E2 == E3 + Ep, this is also given;
Hence, k2x, k2y, k2z can be calculated via momentum conservation.

m1_squared, m2_squared, m3_squared and mp_squared are given;

i_collision specifies which precess this function is going to calculate.

Params
======
flavor, collision_type, i_collision:
    flavor[i_collision,0],flavor[i_collision,1],flavor[i_collision,2],flavor[i_collision,3]
    give the particle species, eg. 0, 1, 0, 1 corresponds to u + d -> u + d
    collision_type[i_collision] gives the type of the collision.
    hence the specific porcess is determined and the amplitude squared is determined.
collision_term:
    array of the same shape as f_x_p_t, 
    collision_term*half_px*half_py*half_pz*half_px*half_py*2/((2*math.pi)**5)/(num_samples*4)
    gives the right handside of the Boltzmann Equation df/dt= C[...]
f_x_p_t:
    array of shape [7, x_grid_size, y_grid_size, z_grid_size, 
                       px_grid_size, py_grid_size, pz_grid_size]
m1_squared, k1x, k1y, k1z, m2_squared, 
m3_squared, k3x, k3y, k3z, E3, mp_squared, px, py, pz, Ep,
half_px, half_py, half_pz,
ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
g, dF, CA, CF, dA, Nq, Ng, mg_regulator_squared, p_type, 

'''

@cuda.jit(device=True)
def accumulate_collision_term(f_x_p_t, m1_squared, k1x, k1y, k1z, m2_squared, 
                              m3_squared, k3x, k3y, k3z, E3, mp_squared, px, py, pz, Ep,
                              half_px, half_py, half_pz,
                              ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                              g, dF, CA, CF, dA, Nq, Ng, mg_regulator_squared, p_type, 
                              flavor, collision_type, i_collision, collision_term, coef):
    result = 0.
    # k1z must be in the box range
    
    if k1z > -half_pz and k1z < half_pz:
        
        # obtain k2x, k2y, k2z via momentum conservation
        k2x = k3x + px - k1x
        k2y = k3y + py - k1y
        k2z = k3z + pz - k1z
        
        # k2x, k2y, k2z must also be in the box range
        if (k2x > -half_px and k2x < half_px and 
            k2y > -half_py and k2y < half_py and 
            k2z > -half_pz and k2z < half_pz):
            
            # energy for particle 1 and 2
            E1, E2= math.sqrt(m1_squared+k1x**2+k1y**2+k1z**2), math.sqrt(m2_squared+k2x**2+k2y**2+k2z**2)

            # jacobin term
            Ja = abs(k1z/E1 - (-k1z+k3z+pz)/E2)
            
            # Ja should be around 1, hence we exclude Ja that are 5 magnitudes smaller
            if Ja>1e-5:
                
                # index of k1,k2,k3
                ik1x, ik1y, ik1z = int((k1x + half_px)//dpx), int((k1y + half_py)//dpy), int((k1z + half_pz)//dpz)
                ik2x, ik2y, ik2z = int((k2x + half_px)//dpx), int((k2y + half_py)//dpy), int((k2z + half_pz)//dpz)
                ik3x, ik3y, ik3z = int((k3x + half_px)//dpx), int((k3y + half_py)//dpy), int((k3z + half_pz)//dpz)
                
                # distribution function: f1,f2,f3,fp
                f = cuda.local.array(shape=4, dtype=float64)
                
                # put index into a collected tuple
                ipx_collect = (ik1x, ik2x, ik3x, ipx)
                ipy_collect = (ik1y, ik2y, ik3y, ipy)
                ipz_collect = (ik1z, ik2z, ik3z, ipz)
                
                # feed values of distribution function and its quantum correction
                tf = cuda.local.array(shape=4, dtype=float64)
                for i_particle in range(4):
                    f[i_particle] = f_x_p_t[flavor[i_collision,i_particle],ix,iy,iz,\
                                                   ipx_collect[i_particle],\
                                                   ipy_collect[i_particle],\
                                                   ipz_collect[i_particle]]
                    if flavor[i_collision,i_particle] == 6:
                        tf[i_particle] = 1 + f[i_particle]
                    else:
                        tf[i_particle] = 1 - f[i_particle]
                
                # if distribution is too small, the collision is of low probability
                if f[0] > 1e-9 and f[1] > 1e-9:
                    # distribution terms      
                    distribution_terms = f[0]*f[1]*tf[2]*tf[3]-tf[0]*tf[1]*f[2]*f[3]

                    # Mandelstam variables s t and u
                    # add a regulator for t and u channels
                    s = (E1+E2)**2-(k1x+k2x)**2-(k1y+k2y)**2-(k1z+k2z)**2
                    t = (E1-E3)**2-(k1x-k3x)**2-(k1y-k3y)**2-(k1z-k3z)**2
                    u = (E1-Ep)**2-(k1x-px)**2-(k1y-py)**2-(k1z-pz)**2

                    # amplitude squared
                    if collision_type[i_collision]==0:
                        # q1+q2->q1+q2 
                        amplitude_square = a_q1q2_q1q2(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)
                    elif collision_type[i_collision]==1:
                        # q+q->q+q  
                        amplitude_square = a_qq_qq(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)
                    elif collision_type[i_collision]==2:
                        # q+q_bar -> q+q_bar
                        amplitude_square = a_qqbar_qqbar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)
                    elif collision_type[i_collision]==3:
                        # q1+q1_bar -> q2+q2_bar
                        amplitude_square = a_q1q1bar_q2q2bar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)
                    elif collision_type[i_collision]==4:
                        # q1+q1_bar -> gg
                        amplitude_square = a_q1q1bar_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)
                    elif collision_type[i_collision]==5:
                        # q+g -> q+g
                        amplitude_square = a_qg_qg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)
                    elif collision_type[i_collision]==6:
                        # gg -> gg
                        amplitude_square = a_gg_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared)
                    
                    # very large amplitude needs to be neglected
                    if amplitude_square < 10**8:
                        # symmetry factor = 0.5 for same incoming particle species 
                        # and 1. for differnent particle species
                        if flavor[i_collision,0]==flavor[i_collision,1]:
                            symmetry_factor = 0.5
                        else:
                            symmetry_factor = 1.0

                        # accumulate collision kernel
                        # some factors are compensated later
                        result = distribution_terms*amplitude_square*symmetry_factor\
                                 *(1/Ja)/(E1*E2*E3*Ep)*coef

                        # save reslut for four times
                        if abs(result)>1e-13:
                            ck_temp = cuda.local.array(shape=4, dtype=float64)
                            for i_ck in range(4):
                                direction = 1.
                                if i_ck < 1.5:
                                    direction = -1.
                                if flavor[i_collision,i_ck]==6: ck_temp[i_ck] = direction*result/Ng
                                else: ck_temp[i_ck] = direction*result/Nq

                            cuda.atomic.add(collision_term, \
                                        (flavor[i_collision,0],ix,iy,iz,ik1x,ik1y,ik1z), ck_temp[0])
                            cuda.atomic.add(collision_term, \
                                        (flavor[i_collision,1],ix,iy,iz,ik2x,ik2y,ik2z), ck_temp[1])
                            cuda.atomic.add(collision_term, \
                                        (flavor[i_collision,2],ix,iy,iz,ik3x,ik3y,ik3z), ck_temp[2])
                            cuda.atomic.add(collision_term, \
                                        (flavor[i_collision,3],ix,iy,iz,ipx,ipy,ipz), ck_temp[3])                     

                        
                    
'''
Amplitude sqaured in the order of Table I in paper: 
J.W. Chen,et al., Phys.Rev.D87, 036002 (2013)

Here we considered the process of:
q1 + q2 -> q1 + q2
q + q -> q + q
q + qbar -> q + qbar
q1 + q1bar -> q2 + q2bar
q1 + q1bar -> g + g
q + g -> q + g
g + g -> g + g

This part can be re-defined according to the specific usage
of users. The argument allowed for this function is limited to:

Params
======
f_x_p_t: 
    input distribution data, 
    shape = [7,x_grid_size,y_grid_size,z_grid_size,
    px_grid_size,py_grid_size,pz_grid_size]
    
k1x, k1y, k1z, m1_squared, E1
k2x, k2y, k2z, m2_squared, E2
k3x, k3y, k3z, m3_squared, E3
px, py, pz, mp_squared, Ep:
    momentum component and the corresponding mass and energy
    
half_px, half_py, half_pz:
    half length of the momentum box
    
ix, iy, iz:
    postional index, this is only used in the indexing of f_x_p_t
    
ik1x, ik1y, ik1z =  int((k1x + half_px)//dpx), \
                    int((k1y + half_py)//dpy), \
                    int((k1z + half_pz)//dpz)
ik2x, ik2y, ik2z =  int((k2x + half_px)//dpx), \
                    int((k2y + half_py)//dpy), \
                    int((k2z + half_pz)//dpz)
ik3x, ik3y, ik3z =  int((k3x + half_px)//dpx), \
                    int((k3y + half_py)//dpy), \
                    int((k3z + half_pz)//dpz)
                    
ipx, ipy, ipz:
    momentum index, this is used for the indexing of f_x_p_t
    
dpx, dpy, dpz:
    grid length for momentum box
    
g, dF, CA, CF, dA, Nq, Ng:
    constants that can be used
    
mg_regulator_squared:
    mg_regulator_squared is taken to be the gluon thermal mass
    
f[0],f[1],tf[2],tf[3],tf[0],tf[1],f[2],f[3]:
    distribution for the current process. eg:
    f[0] = f_x_p_t[p_type,ix,iy,iz,ipx,ipy,ipz]
    
# Mandelstam variables s t and u
s = (E1+E2)**2-(k1x+k2x)**2-(k1y+k2y)**2-(k1z+k2z)**2
t = (E1-E3)**2-(k1x-k3x)**2-(k1y-k3y)**2-(k1z-k3z)**2
u = (E1-Ep)**2-(k1x-px)**2-(k1y-py)**2-(k1z-pz)**2
<<<

A regulator in the denominator is added here to prevent divergence.

'''


@cuda.jit(device=True)
def a_q1q2_q1q2(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
    return 8*(g**4)*(dF**2)*(CF**2)/dA*((s*s+u*u)/(t-mg_regulator_squared)**2)
        
@cuda.jit(device=True)
def a_qq_qq(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
    return 8*(g**4)*(dF**2)*(CF**2)/dA*((s*s+u*u)/((t-mg_regulator_squared)**2)+\
                                         (s*s+t*t)/((u-mg_regulator_squared)**2))+\
           16*(g**4)*dF*CF*(CF-CA/2)*(s*s)/((t-mg_regulator_squared)*(u-mg_regulator_squared))
    
@cuda.jit(device=True)
def a_qqbar_qqbar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
    return 8*(g**4)*(dF**2)*(CF**2)/dA*((s*s+u*u)/((t-mg_regulator_squared)**2)+\
                                         (u*u+t*t)/(s*s))+\
           16*(g**4)*dF*CF*(CF-CA/2)*(u*u)/((t-mg_regulator_squared)*s)
    
@cuda.jit(device=True)
def a_q1q1bar_q2q2bar(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
    return 8*(g**4)*(dF**2)*(CF**2)/dA*((t*t+u*u)/(s*s))
    
@cuda.jit(device=True)
def a_q1q1bar_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
    return 8*g**4*dF*CF**2*(u/(t-mg_regulator_squared)+t/(u-mg_regulator_squared))-\
           8*g**4*dF*CF*CA*(t**2+u**2)/(s**2)
    
@cuda.jit(device=True)
def a_qg_qg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
    return -8*g**4*dF*CF**2*(u/s+s/(u-mg_regulator_squared))+\
            8*g**4*dF*CF*CA*(s*s+u*u)/((t-mg_regulator_squared)**2)
    
@cuda.jit(device=True)
def a_gg_gg(g,s,t,u,dF,CA,CF,dA,mg_regulator_squared):
    return 16*g**4*dA*CA**2*(3-s*u/((t-mg_regulator_squared)**2)
                             -s*t/((u-mg_regulator_squared)**2)
                             -t*u/(s**2))

'''
Device function to find k1_r from E1 + E2 == E3 + Ep.
The general solution of k1x has two forms:
k1x = (C1+math.sqrt(H))/C2 or (C1-math.sqrt(H))/C2.

Note that H can be negative and C2 can be zero, 
therefore a check is needed to make sure 
that the program is working properly without any
mathematical domain error.

The general form is found via Mathematica.

Params
======
m1Squared:
    masses squared of particle 1
        k1x,k1y:
    momentum of particle 1
m2Squared:
    masses squared of particle 2
E3,k3x,k3y,k3z:
    energy and momentum for particle 3
Ep,px,py,pz:
    energy and momentum for particle p
'''

@cuda.jit(device=True)
def H_fun(m1Squared,k1x,k1y,m2Squared,E3,k3x,k3y,k3z,Ep,px,py,pz):
    return (E3 + Ep)**2*(E3**4 + 4*E3**3*Ep + Ep**4 + 4*k1x**2*k3x**2 - 4*k1x*k3x**3 + k3x**4 + 8*k1x*k1y*k3x*k3y - 4*k1y*k3x**2*k3y \
                         + 4*k1y**2*k3y**2 - 4*k1x*k3x*k3y**2 + 2*k3x**2*k3y**2 - 4*k1y*k3y**3 + k3y**4 + 4*k1x**2*k3z**2 + 4*k1y**2*k3z**2 \
                         - 4*k1x*k3x*k3z**2 + 2*k3x**2*k3z**2 - 4*k1y*k3y*k3z**2 + 2*k3y**2*k3z**2 + k3z**4 + 4*k1x*k3x*m1Squared \
                         - 2*k3x**2*m1Squared + 4*k1y*k3y*m1Squared - 2*k3y**2*m1Squared + 2*k3z**2*m1Squared + m1Squared**2 \
                         - 4*k1x*k3x*m2Squared + 2*k3x**2*m2Squared - 4*k1y*k3y*m2Squared + 2*k3y**2*m2Squared + 2*k3z**2*m2Squared \
                         - 2*m1Squared*m2Squared + m2Squared**2 + 8*k1x**2*k3x*px - 12*k1x*k3x**2*px + 4*k3x**3*px + 8*k1x*k1y*k3y*px \
                         - 8*k1y*k3x*k3y*px - 4*k1x*k3y**2*px + 4*k3x*k3y**2*px - 4*k1x*k3z**2*px + 4*k3x*k3z**2*px + 4*k1x*m1Squared*px \
                         - 4*k3x*m1Squared*px - 4*k1x*m2Squared*px + 4*k3x*m2Squared*px + 4*k1x**2*px**2 - 12*k1x*k3x*px**2 + 6*k3x**2*px**2 \
                         - 4*k1y*k3y*px**2 + 2*k3y**2*px**2 + 2*k3z**2*px**2 - 2*m1Squared*px**2 + 2*m2Squared*px**2 - 4*k1x*px**3 + 4*k3x*px**3 \
                         + px**4 + 8*k1x*k1y*k3x*py - 4*k1y*k3x**2*py + 8*k1y**2*k3y*py - 8*k1x*k3x*k3y*py + 4*k3x**2*k3y*py - 12*k1y*k3y**2*py \
                         + 4*k3y**3*py - 4*k1y*k3z**2*py + 4*k3y*k3z**2*py + 4*k1y*m1Squared*py - 4*k3y*m1Squared*py - 4*k1y*m2Squared*py \
                         + 4*k3y*m2Squared*py + 8*k1x*k1y*px*py - 8*k1y*k3x*px*py - 8*k1x*k3y*px*py + 8*k3x*k3y*px*py - 4*k1y*px**2*py \
                         + 4*k3y*px**2*py + 4*k1y**2*py**2 - 4*k1x*k3x*py**2 + 2*k3x**2*py**2 - 12*k1y*k3y*py**2 + 6*k3y**2*py**2 \
                         + 2*k3z**2*py**2 - 2*m1Squared*py**2 + 2*m2Squared*py**2 - 4*k1x*px*py**2 + 4*k3x*px*py**2 + 2*px**2*py**2 \
                         - 4*k1y*py**3 + 4*k3y*py**3 + py**4 \
                         + 4*k3z*(2*k1x**2 + 2*k1y**2 + k3z**2 + m1Squared + m2Squared - 2*k1x*(k3x + px) + (k3x + px)**2 - 2*k1y*(k3y + py) \
                                  + (k3y + py)**2)*pz + 2*(2*k1x**2 + 2*k1y**2 + 3*k3z**2 + m1Squared + m2Squared - 2*k1x*(k3x + px) \
                                                           + (k3x + px)**2 - 2*k1y*(k3y + py) + (k3y + py)**2)*pz**2 + 4*k3z*pz**3 \
                         + pz**4 + 4*E3*Ep*(Ep**2 - 2*k1x**2 - 2*k1y**2 - m1Squared - m2Squared + 2*k1x*(k3x + px) - (k3x + px)**2 \
                                            + 2*k1y*(k3y + py) - (k3y + py)**2 - (k3z + pz)**2) \
                         - 2*Ep**2*(2*k1x**2 + 2*k1y**2 + m1Squared + m2Squared - 2*k1x*(k3x + px) \
                                    + (k3x + px)**2 - 2*k1y*(k3y + py) + (k3y + py)**2 + (k3z + pz)**2) \
                         + E3**2*(6*Ep**2 - 2*(2*k1x**2 + 2*k1y**2 + m1Squared + m2Squared - 2*k1x*(k3x + px) \
                                               + (k3x + px)**2 - 2*k1y*(k3y + py) + (k3y + py)**2 + (k3z + pz)**2)))

@cuda.jit(device=True)
def C1_fun(m1Squared,k1x,k1y,m2Squared,E3,k3x,k3y,k3z,Ep,px,py,pz):
    return (-k3z - pz)*(-(E3 + Ep)**2 - m1Squared + m2Squared - 2*k1x*(k3x + px) + (k3x + px)**2 - 2*k1y*(k3y + py) + (k3y + py)**2 + (k3z + pz)**2)

@cuda.jit(device=True)
def C2_fun(m1Squared,k1x,k1y,m2Squared,E3,k3x,k3y,k3z,Ep,px,py,pz):
    return 2*(E3 + Ep - k3z - pz)*(E3 + Ep + k3z + pz)

'''
Device function to specify px, py, pz, k1x, k1y, k1z, k3x, k3y, k3z.

k3x, k3y, k3z, k1x, k1y are sampled randomly in the entire
momentum box;

px, py, pz are sampled randomly in the small momentum grid;
Note that this random sample is different from the sample
of k3x, k3y, k3z, k1x, k1y. The sample of p is in the
specific momentum grid, which corresponds to the scan in the left handside
of Boltzmann Equation df(p,x,t)/dt = ... While the sample of k3 and k1 is in
the entire momentum box which corresponds to the direct Monte Carlo integration.

k1z is obtained by solving E1 + E2 == E3 + Ep with two possible values.

Params
======
f_x_p_t:
    array of shape [7, x_grid_size, y_grid_size, z_grid_size, 
                       px_grid_size, py_grid_size, pz_grid_size]
flavor, collision_type:
    flavor[i_collision,0],flavor[i_collision,1],flavor[i_collision,2],flavor[i_collision,3]
    give the particle species, eg. 0, 1, 0, 1 corresponds to u + d -> u + d
    collision_type[i_collision] gives the type of the collision.
masses_squared_collect:
    array of shape 4, each element corresponds to the mass squared of the current 
    GPU thread (phase grid). 
    [mu^2, md^2, ms^2, mg^2]
num_samples:
    int, number of samples that should be sampled for direct Monte Carlo integration.
collision_term:
    array of the same shape as f_x_p_t, 
    collision_term*half_px*half_py*half_pz*half_px*half_py*2/((2*math.pi)**5)/(num_samples*4)
    gives the right handside of the Boltzmann Equation df/dt= C[...]
p_type:
    int, corresponds to the particle species of the fourth particle.
mg_regulator_squared:
    takes the same value of mg^2
half_px, half_py, half_pz,
    rng_states, real_id, mp_squared,
ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
g, dF, CA, CF, dA, Nq, Ng

'''


@cuda.jit(device=True)
def collision_term_at_specific_point(f_x_p_t, flavor, collision_type,
                                     masses, num_samples,
                                     half_px, half_py, half_pz,
                                     rng_states, thread_id, mp_squared,
                                     ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                                     p_type, degeneracy,
                                     collision_term, px, py, pz, Ep, coef,
                                     npx,npy,npz,nx,ny,nz):
    
    result = 0.
    # loop through all possible collision types
    for i_collision in range(len(collision_type)):
        
        # exclude the collision types that are non-exist
        # 100 indicates that the maximum number of particles supported is 100
        if collision_type[i_collision] < 100:
            # particle masses at this spatial point
            m1_squared = masses[flavor[i_collision,0]]**2
            m2_squared = masses[flavor[i_collision,1]]**2
            m3_squared = masses[flavor[i_collision,2]]**2

            # Direct Monte Carlo Method to accumulate point values
            for i_sample in range(num_samples):

                # randomly generate k1x, k1y, k3x, k3y, k3z
                k1x = (xoroshiro128p_uniform_float64(rng_states, thread_id)*2-1)*half_px
                k1y = (xoroshiro128p_uniform_float64(rng_states, thread_id)*2-1)*half_py
                k3x = (xoroshiro128p_uniform_float64(rng_states, thread_id)*2-1)*half_px
                k3y = (xoroshiro128p_uniform_float64(rng_states, thread_id)*2-1)*half_py
                k3z = (xoroshiro128p_uniform_float64(rng_states, thread_id)*2-1)*half_pz

                E3= math.sqrt(m3_squared+k3x**2+k3y**2+k3z**2)

                # solving for k1z via energy conservation
                # k1z has two values
                C1 = C1_fun(m1_squared,k1x,k1y,m2_squared,E3,k3x,k3y,k3z,Ep,px,py,pz)
                C2 = C2_fun(m1_squared,k1x,k1y,m2_squared,E3,k3x,k3y,k3z,Ep,px,py,pz)
                H = H_fun(m1_squared,k1x,k1y,m2_squared,E3,k3x,k3y,k3z,Ep,px,py,pz)

                # take the value if H>1e-13 and abs(C2)>1e-13
                if H>1e-13 and abs(C2)>1e-13:
                    k1z1, k1z2 = (C1+math.sqrt(H))/C2, (C1-math.sqrt(H))/C2

                    # call accumulate_collision_term for the two values
                    for k1z in (k1z1,k1z2):

                        accumulate_collision_term(f_x_p_t, m1_squared, k1x, k1y, k1z, m2_squared, 
                                                  m3_squared, k3x, k3y, k3z, E3, mp_squared, px, py, pz, Ep,
                                                  half_px, half_py, half_pz,
                                                  ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                                                  p_type, degeneracy,
                                                  flavor, collision_type, i_collision, collision_term, coef,
                                                  npx,npy,npz,nx,ny,nz)
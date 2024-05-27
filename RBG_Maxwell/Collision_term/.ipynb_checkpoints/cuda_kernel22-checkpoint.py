from numba import float64
import math
from numba import cuda        
from numba.cuda.random import xoroshiro128p_uniform_float64
from .. import Collision_database

# import the correct amplitude
with open(Collision_database.__path__[0]+'/selected_system.txt','r') as save_text:
    string = 'from ..Collision_database.selected_system.amplitude_square22 import Amplitude_square'.replace("selected_system",save_text.read()) 
    exec(string)

@cuda.jit(device=True)
def threeD_to_oneD(ix, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz):
    return ipz+ipy*npz+ipx*npz*npy+iz*npz*npy*npx+iy*npz*npy*npx*nz+ix*npz*npy*npx*nz*ny
    
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

k1z is obtained by solving k10 + k20 == k30 + p0, this is also given;
Hence, k2x, k2y, k2z can be calculated via momentum conservation.

m1_squared, m2_squared, m3_squared and mp_squared are given;

i_collision specifies which precess this function is going to calculate.

'''
                                                

@cuda.jit(device=True)
def accumulate_collision_term(f_x_p_t0level, f_x_p_tilevel, m1_squared, k1x, k1y, k1z, m2_squared, 
                              m3_squared, k3x, k3y, k3z, k30, mp_squared, px, py, pz, p0,
                              half_px, half_py, half_pz, dp, masses,
                              ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                              p_type, i_grid, particle_type, degeneracy,
                              flavor, collision_type, 
                              i_collision, 
                              collision_term0level, collision_termilevel,
                              npx, npy, npz, nx, ny, nz, i_level,
                              num_momentum_level, flavor0, flavor1, flavor2,
                              middle_npx, middle_npy, middle_npz,
                              int_volume, hbar,c,lambdax):
    # k1z must be in the momentum box range of particle type 0 
    if k1z > -half_pz[flavor0] and k1z < half_pz[flavor0]:
        
        # obtain k2x, k2y, k2z via momentum conservation
        k2x = k3x + px - k1x
        k2y = k3y + py - k1y
        k2z = k3z + pz - k1z
        
        # k2x, k2y, k2z must also be in the momentum box range of particle 1
        if (k2x > -half_px[flavor1] and k2x < half_px[flavor1] and 
            k2y > -half_py[flavor1] and k2y < half_py[flavor1] and 
            k2z > -half_pz[flavor1] and k2z < half_pz[flavor1]):
            
            # energy for particle 1 and 2
            k10, k20 = math.sqrt(m1_squared*c**2+k1x**2+k1y**2+k1z**2), math.sqrt(m2_squared*c**2+k2x**2+k2y**2+k2z**2)

            # energy must be conserved
            if abs(k10+k20-k30-p0) < 10**(-18):
                # jacobin term
                Ja = abs(k1z/k10 - (-k1z+k3z+pz)/k20)
                # momentum product
                momentum_product = k10*k20*k30*p0

                # momentum_producted cannot be zero
                # Ja should be around 1, hence we exclude Ja that are 5 magnitudes smaller
                if Ja>1e-5 and momentum_product > 10**(-19):
                    ################################################################################
                    ################################################################################
                    # for particle 0
                    ik1x, ik1y, ik1z = int((k1x + half_px[flavor0])//dpx[flavor0]), \
                                       int((k1y + half_py[flavor0])//dpy[flavor0]), \
                                       int((k1z + half_pz[flavor0])//dpz[flavor0])
                    # convert grid index in one dimension
                    i_phase_grid0 = threeD_to_oneD(ix,iy,iz,ik1x,ik1y,ik1z, \
                                                   nx, ny, nz, npx, npy, npz)

                    ################################################################################ 
                    ################################################################################
                    # for particle 1
                    ik2x, ik2y, ik2z = int((k2x + half_px[flavor1])//dpx[flavor1]), \
                                       int((k2y + half_py[flavor1])//dpy[flavor1]), \
                                       int((k2z + half_pz[flavor1])//dpz[flavor1])
                    # convert grid index in one dimension
                    i_phase_grid1 = threeD_to_oneD(ix,iy,iz,ik2x,ik2y,ik2z, \
                                                   nx, ny, nz, npx, npy, npz)

                    ################################################################################
                    ################################################################################
                    # for particle 2
                    ik3x, ik3y, ik3z = int((k3x + half_px[flavor2])//dpx[flavor2]), \
                                       int((k3y + half_py[flavor2])//dpy[flavor2]), \
                                       int((k3z + half_pz[flavor2])//dpz[flavor2])
                    # convert grid index in one dimension
                    i_phase_grid2 = threeD_to_oneD(ix,iy,iz,ik3x,ik3y,ik3z, \
                                                   nx, ny, nz, npx, npy, npz)

                    ################################################################################
                    ################################################################################

                    # distribution function: f1,f2,f3,fp
                    f = cuda.local.array(shape=4, dtype=float64)
                    f[0] = f_x_p_t0level[flavor0, i_phase_grid0]
                    f[1] = f_x_p_t0level[flavor1, i_phase_grid1]
                    f[2] = f_x_p_t0level[flavor2, i_phase_grid2]
                    f[3] = f_x_p_tilevel[p_type, i_grid]

                    # feed values of distribution function and its quantum correction
                    tf = cuda.local.array(shape=4, dtype=float64)
                    # particle_type: 0, 1, 2 for classical, fermi and Bosonic
                    for i_particle in range(4):
                        i_flavor = flavor[i_collision,i_particle]
                        if particle_type[i_flavor] == 1:
                            tf[i_particle] = 1 - (2*math.pi*hbar)**3*f[i_particle]/degeneracy[i_flavor]
                        elif particle_type[i_flavor] == 2:
                            tf[i_particle] = 1 + (2*math.pi*hbar)**3*f[i_particle]/degeneracy[i_flavor]
                        else:
                            tf[i_particle] = 1

                    # initial and finalsummed amplitude square
                    amplitude_square = Amplitude_square(m1_squared,m2_squared,m3_squared,mp_squared,\
                                                        k10,k20,k30,p0,collision_type[i_collision],\
                                                        k1x,k1y,k1z,k2x,k2y,k2z,k3x,k3y,k3z,\
                                                        px,py,pz,hbar,c,lambdax)

                    # distribution with amplitude squared (initial and final summed)
                    #  to be consistent with the requirement in paper that M_squared must be initial averaged and final summed, we divide the relevant degenaracies here
                    # f1*f2*tf3*tfp - tf1*tf2*f3*fp
                    distribution_terms = f[0]*f[1]*tf[2]*tf[3]*amplitude_square/degeneracy[flavor[i_collision,0]]/degeneracy[flavor[i_collision,1]] - tf[0]*tf[1]*f[2]*f[3]*amplitude_square/degeneracy[flavor[i_collision,2]]/degeneracy[flavor[i_collision,3]]

                    # symmetry factor = 0.5 for same incoming particle species 
                    # and 1. for differnent particle species
                    if flavor0==flavor1:
                        symmetry_factor = 0.5
                    else:
                        symmetry_factor = 1.0

                    # accumulate collision kernel
                    # some factors are compensated later
                    result = distribution_terms*symmetry_factor\
                           *hbar**2*c/Ja*int_volume/(momentum_product*64*math.pi**2)

                    cuda.atomic.add(collision_term0level, \
                                (flavor0, i_phase_grid0), -result/dp[flavor0]*dp[p_type])
                    cuda.atomic.add(collision_term0level, \
                                (flavor1, i_phase_grid1), -result/dp[flavor1]*dp[p_type])
                    cuda.atomic.add(collision_term0level, \
                                (flavor2, i_phase_grid2), result/dp[flavor2]*dp[p_type])
                    cuda.atomic.add(collision_termilevel, (p_type, i_grid), result)

                            
'''
Device function to find k1_r from E1 + E2 == E3 + Ep.
The general solution of k1x has two forms:
k1x = (C1+math.sqrt(H))/C2 or (C1-math.sqrt(H))/C2.

Note that H can be negative and C2 can be zero, 
therefore a check is needed to make sure 
that the program is working properly without any
mathematical domain error.

The general form is found via Mathematica.
'''

@cuda.jit(device=True)
def H_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,p0,px,py,pz,c):
    return (k30 + p0)**2*\
(k30**4 - 2*k30**2*k3x**2 + k3x**4 - 2*k30**2*k3y**2 + \
2*k3x**2*k3y**2 + k3y**4 - 2*k30**2*k3z**2 + \
2*k3x**2*k3z**2 + 2*k3y**2*k3z**2 + k3z**4 - \
2*c**2*k30**2*m1**2 - 2*c**2*k3x**2*m1**2 - \
2*c**2*k3y**2*m1**2 + 2*c**2*k3z**2*m1**2 + \
c**4*m1**4 - 2*c**2*k30**2*m2**2 + \
2*c**2*k3x**2*m2**2 + 2*c**2*k3y**2*m2**2 + \
2*c**2*k3z**2*m2**2 - 2*c**4*m1**2*m2**2 + c**4*m2**4 + \
4*k30**3*p0 - 4*k30*k3x**2*p0 - 4*k30*k3y**2*p0 - \
4*k30*k3z**2*p0 - 4*c**2*k30*m1**2*p0 - \
4*c**2*k30*m2**2*p0 + 6*k30**2*p0**2 - 2*k3x**2*p0**2 - \
2*k3y**2*p0**2 - 2*k3z**2*p0**2 - 2*c**2*m1**2*p0**2 - \
2*c**2*m2**2*p0**2 + 4*k30*p0**3 + p0**4 - \
4*k30**2*k3x*px + 4*k3x**3*px + 4*k3x*k3y**2*px + \
4*k3x*k3z**2*px - 4*c**2*k3x*m1**2*px + \
4*c**2*k3x*m2**2*px - 8*k30*k3x*p0*px - \
4*k3x*p0**2*px - 2*k30**2*px**2 + 6*k3x**2*px**2 + \
2*k3y**2*px**2 + 2*k3z**2*px**2 - 2*c**2*m1**2*px**2 + \
2*c**2*m2**2*px**2 - 4*k30*p0*px**2 - 2*p0**2*px**2 + \
4*k3x*px**3 + px**4 - 4*k30**2*k3y*py + \
4*k3x**2*k3y*py + 4*k3y**3*py + 4*k3y*k3z**2*py - \
4*c**2*k3y*m1**2*py + 4*c**2*k3y*m2**2*py - \
8*k30*k3y*p0*py - 4*k3y*p0**2*py + 8*k3x*k3y*px*py + \
4*k3y*px**2*py - 2*k30**2*py**2 + 2*k3x**2*py**2 + \
6*k3y**2*py**2 + 2*k3z**2*py**2 - 2*c**2*m1**2*py**2 + \
2*c**2*m2**2*py**2 - 4*k30*p0*py**2 - 2*p0**2*py**2 + \
4*k3x*px*py**2 + 2*px**2*py**2 + 4*k3y*py**3 + py**4 + \
4*k3z*(k3z**2 + c**2*(m1**2 + m2**2) - (k30 + p0)**2 + \
(k3x + px)**2 + (k3y + py)**2)*pz + \
2*(3*k3z**2 + c**2*(m1**2 + m2**2) - (k30 + p0)**2 + \
(k3x + px)**2 + (k3y + py)**2)*pz**2 + 4*k3z*pz**3 + \
pz**4 + 4*k1x**2*\
(-(k30 + p0)**2 + (k3x + px)**2 + (k3z + pz)**2) + \
4*k1y**2*(-(k30 + p0)**2 + (k3y + py)**2 + \
(k3z + pz)**2) - \
4*k1y*(k3y + py)*\
(c**2*(-m1**2 + m2**2) - (k30 + p0)**2 + \
(k3x + px)**2 + (k3y + py)**2 + (k3z + pz)**2) - \
4*k1x*(k3x + px)*\
(c**2*(-m1**2 + m2**2) - (k30 + p0)**2 + \
(k3x + px)**2 - 2*k1y*(k3y + py) + (k3y + py)**2 + \
(k3z + pz)**2))

@cuda.jit(device=True)
def C1_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,p0,px,py,pz,c):
    return -((k3z + pz)*\
(c**2*(-m1**2 + m2**2) - (k30 + p0)**2 -\
2*k1x*(k3x + px) + (k3x + px)**2 - 2*k1y*(k3y + py) + \
(k3y + py)**2 + (k3z + pz)**2))

@cuda.jit(device=True)
def C2_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,p0,px,py,pz,c):
    return (2.*(k30 - k3z + p0 - pz)*(k30 + k3z + p0 + pz))

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

params
======
f_x_p_t0level: 
    distributions of the 0-th momentum level
    of size [n_type, nx, ny, nz, npx, npy, npz]
f_x_p_tilevel: 
    distributions of the i-th momentum level
    of size [n_type, nx, ny, nz, npx, npy, npz]
flavor and collision_type:
    numpy arrays.

    # the principle for writing the possible collisions is that the collisions
    # must be symmetrical, i.e., if we have a + b -> c + d, we must at the same time
    # consider c + d -> a + b. Note that a + a -> a + a is symmetrical already.

    flavor and collision_type:
            numpy arrays.
            
            # flavor: all possible collisions for the given final particle, eg: 
            #        for final d, we have
            #        ud->ud (0), dd->dd (1), sd->sd (0), u_bar+d->u_bar+d (0), s_bar+d->s_bar+d (0), gd->gd (5)
            #        d_bar+d->d_bar+d (2), u_bar+u->d_bar+d (3), s_bar+s->d_bar+d (3), gg->d_bar+d (4)
            
            The corresponding flavor array is
            #        flavor=np.array([[[0,1,0,1],[1,1,1,1],[2,1,2,1],[3,1,3,1],[5,1,5,1],[6,1,6,1],\
            #                         [4,1,4,1],[3,0,4,1],[5,2,4,1],[6,6,4,1]]],dtype=np.int64)
            
            # collision_type: an index indicate which collision type the process belongs to, eg:
            For final d quark case
            #                collision_type=np.array([[0,1,0,0,0,5,2,3,3,4]],dtype=np.int64)
            
            where 0,1,2,3,4,5,6 corresponds to the following processes:
            (0): q1 + q2 -> q1 + q2
            (1): q + q -> q + q
            (2): q + qbar -> q + qbar
            (3): q1 + q1bar -> q2 + q2bar
            (4): q1 + q1bar -> g + g
            (5): q + g -> q + g
            (6): g + g -> g + g
particle_type:
    particle types correspond to different particles, 0 for classical, 
    1 for fermion, and 2 for bosonic, e.g., [1,1,1]
degeneracy:
    degenacies for particles, for quarks this is 6, e.g., [6,6,6,6,6,6]
masses:
    masses for each type of particles, e.g., [0.2, 0.2] 
num_samples:
    int, number of samples for the five dimensional integration for collision term.
half_px, half_py, half_pz:
    negative left boundary of the momentum region, 
    this is an array containing the values for different type of particles 
rng_states:
    array for generate random samples.    
mp_squared:
    mass squared of particle p

num_momentum_level:
    number of straitified momentum levels 
total_grid:
    total number of grids, nx*ny*nz*npx*npy*npz
num_of_particle_types:
    total number of particle types, this is len(masses)
npx,npy,npz: 
    number of grid sizes in in momentum domain, e.g., [5,5,5]
nx,ny,nz:
    number of grids in spatial domain, e.g., [5, 5, 5]
             
dpx, dpy, dpz:
    infinitesimal difference in momentum coordinate, 
    this is an array containing the values for different type of particles
dp:
    dpx*dpy*dpz

collision_term0level, collision_termilevel
    this is the term that needs to be evaluated, of the same size as f_x_p_t
middle_npx, middle_npy, middle_npz:
    the middle index of npx, npy, npz
i_level:
    the i-th momentum level
hbar,c,lambdax:
    numerical value of hbar,c,and lambdax in FU

'''


@cuda.jit(device=True)
def collision_term_at_specific_point(f_x_p_t0level, f_x_p_tilevel, 
                                     flavor, collision_type, particle_type, degeneracy,
                                     masses, num_samples,
                                     half_px, half_py, half_pz,
                                     rng_states, mp_squared,
                                     ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                                     p_type, dp,
                                     collision_term0level, collision_termilevel, 
                                     px, py, pz, p0,
                                     npx, npy, npz, nx, ny, nz, i_level,
                                     num_momentum_level, i_grid,
                                     middle_npx, middle_npy, middle_npz,
                                     hbar,c,lambdax):
    
    # loop through all possible collision types
    for i_collision in range(len(collision_type)):
        
        # exclude the collision types that are non-exist
        # 100 indicates that the maximum number of collision types supported is 1000
        if collision_type[i_collision] < 1000:
            
            # particle types for incoming and out going particles
            flavor0 = flavor[i_collision,0]
            flavor1 = flavor[i_collision,1]
            flavor2 = flavor[i_collision,2]
            
            # integration_volume
            # we divide the integration volume by num_samples*4
            # 4 here indicates that the integrations are evaluated for four times due to the symmetric smapling
            # number of samples are for the Monte Carlo integration
            # different particles have different integration domain
            int_volume = 2**5*half_px[flavor0]*half_py[flavor0]*half_px[flavor2]*half_py[flavor2]\
                        *half_pz[flavor2]/(num_samples*4)
            
            # particle masses at this spatial point
            m1 = masses[flavor0]
            m2 = masses[flavor1]
            m3 = masses[flavor2]

            # Direct Monte Carlo Method to accumulate point values
            for i_sample in range(num_samples):
                
                # randomly generate k1x, k1y, k3x, k3y, k3z
                # the momentum values need to be sampled according to the specific particle types
                k1x = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_px[flavor0]
                k1y = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_py[flavor0]
                k3x = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_px[flavor2]
                k3y = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_py[flavor2]
                k3z = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_pz[flavor2]
                
                k30 = math.sqrt(m3**2*c**2+k3x**2+k3y**2+k3z**2)

                # solving for k1z via energy conservation
                # k1z has two values
                C1 = C1_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,p0,px,py,pz,c)
                C2 = C2_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,p0,px,py,pz,c)
                H = H_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,p0,px,py,pz,c)
                # take the value if H>1e-13 and abs(C2)>1e-13
                if H>1e-13 and abs(C2)>1e-13:
                    k1z1, k1z2 = (C1+math.sqrt(H))/C2, (C1-math.sqrt(H))/C2
                   
                    # call accumulate_collision_term for the two values
                    for k1z in (k1z1,k1z2):

                        accumulate_collision_term(f_x_p_t0level, f_x_p_tilevel, m1**2, k1x, k1y, k1z, m2**2, 
                                                  m3**2, k3x, k3y, k3z, k30, mp_squared, px, py, pz, p0,
                                                  half_px, half_py, half_pz, dp, masses,
                                                  ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                                                  p_type, i_grid, particle_type, degeneracy,
                                                  flavor, collision_type, 
                                                  i_collision, 
                                                  collision_term0level, collision_termilevel,
                                                  npx, npy, npz, nx, ny, nz, i_level,
                                                  num_momentum_level, flavor0, flavor1, flavor2,
                                                  middle_npx, middle_npy, middle_npz, int_volume,
                                                  hbar,c,lambdax)




'''
CUDA kernel for obtaining the collision term using Monte Carlo integration.

params
======
f_x_p_t0level: 
    distributions of the 0-th momentum level
    of size [n_type, nx, ny, nz, npx, npy, npz]
f_x_p_tilevel: 
    distributions of the i-th momentum level
    of size [n_type, nx, ny, nz, npx, npy, npz]
masses:
    masses for each type of particles, e.g., [0.2, 0.2] 
num_momentum_level:
    number of straitified momentum levels 
total_grid:
    total number of grids, nx*ny*nz*npx*npy*npz
num_of_particle_types:
    total number of particle types
npx,npy,npz: 
    number of grid sizes in in momentum domain, e.g., [5,5,5]
nx,ny,nz:
    number of grids in spatial domain, e.g., [5, 5, 5]
half_px, half_py, half_pz:
    negative left boundary of the momentum region, 
    this is an array containing the values for different type of particles              
dpx, dpy, dpz:
    infinitesimal difference in momentum coordinate, 
    this is an array containing the values for different type of particles
dp:
    dpx*dpy*dpz
flavor and collision_type:
    numpy arrays.

    # the principle for writing the possible collisions is that the collisions
    # must be symmetrical, i.e., if we have a + b -> c + d, we must at the same time
    # consider c + d -> a + b. Note that a + a -> a + a is symmetrical already.

    flavor and collision_type:
            numpy arrays.
            
            # flavor: all possible collisions for the given final particle, eg: 
            #        for final d, we have
            #        ud->ud (0), dd->dd (1), sd->sd (0), u_bar+d->u_bar+d (0), s_bar+d->s_bar+d (0), gd->gd (5)
            #        d_bar+d->d_bar+d (2), u_bar+u->d_bar+d (3), s_bar+s->d_bar+d (3), gg->d_bar+d (4)
            
            The corresponding flavor array is
            #        flavor=np.array([[[0,1,0,1],[1,1,1,1],[2,1,2,1],[3,1,3,1],[5,1,5,1],[6,1,6,1],\
            #                         [4,1,4,1],[3,0,4,1],[5,2,4,1],[6,6,4,1]]],dtype=np.int64)
            
            # collision_type: an index indicate which collision type the process belongs to, eg:
            For final d quark case
            #                collision_type=np.array([[0,1,0,0,0,5,2,3,3,4]],dtype=np.int64)
            
            where 0,1,2,3,4,5,6 corresponds to the following processes:
            (0): q1 + q2 -> q1 + q2
            (1): q + q -> q + q
            (2): q + qbar -> q + qbar
            (3): q1 + q1bar -> q2 + q2bar
            (4): q1 + q1bar -> g + g
            (5): q + g -> q + g
            (6): g + g -> g + g
particle_type:
    particle types correspond to different particles, 0 for classical, 
    1 for fermion, and 2 for bosonic, e.g., [1,1,1]
degeneracy:
    degenacies for particles, for quarks this is 6, e.g., [6,6,6,6,6,6]
num_samples:
    int, number of samples for the five dimensional integration for collision term.
rng_states:
    array for generate random samples.
collision_term0level, collision_termilevel
    this is the term that needs to be evaluated, of the same size as f_x_p_t
middle_npx, middle_npy, middle_npz:
    the middle index of npx, npy, npz
i_level:
    the i-th momentum level
hbar, c, lambdax:
    numerical value of hbar, c, and lambdax in FU
'''
                          


@cuda.jit
def collision_term_kernel(f_x_p_t0level, f_x_p_tilevel, masses, num_momentum_level,\
                  total_grid, num_of_particle_types, particle_type, degeneracy,\
                  npx,npy,npz,nx,ny,nz, \
                  half_px, half_py, half_pz, \
                  dpx, dpy, dpz, dp,\
                  flavor, collision_type, \
                  num_samples, rng_states,\
                  collision_term0level, collision_termilevel,\
                  middle_npx, middle_npy, middle_npz, i_level,\
                  hbar, c, lambdax):
    
    # threads loop in one dimension
    i_grid = cuda.grid(1)
            
    if i_grid < total_grid:
        
        # convert one-d index into six-d
        ipz = i_grid%npz
        ipz_rest = i_grid//npz
        ipy = ipz_rest%npy
        ipy_rest = ipz_rest//npy
        ipx = ipy_rest%npx
        ipx_rest = ipy_rest//npx
        iz = ipx_rest%nz
        iz_rest = ipx_rest//nz
        iy = iz_rest%ny
        iy_rest = iz_rest//ny
        ix = iy_rest%nx
        
        # loop through all species
        for p_type in range(num_of_particle_types):
            
            # acquire p from the central value
            # Note that for different particles, they have different dpx and px_left_bound
            # the momentum level corresponds to the level of straitification
            px = ((ipx+0.5)*dpx[p_type] - half_px[p_type])/(npx**i_level)
            py = ((ipy+0.5)*dpy[p_type] - half_py[p_type])/(npy**i_level)
            pz = ((ipz+0.5)*dpz[p_type] - half_pz[p_type])/(npz**i_level)

            # p0 for current grid
            mp_squared = masses[p_type]**2
            p0 = math.sqrt(mp_squared*c**2+px**2+py**2+pz**2)

            # collision term
            # evaluate collision term  at certain phase point
            collision_term_at_specific_point(f_x_p_t0level, f_x_p_tilevel,
                                             flavor[p_type], collision_type[p_type],  
                                             particle_type, degeneracy,
                                             masses, num_samples,
                                             half_px, half_py, half_pz,
                                             rng_states, mp_squared,
                                             ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                                             p_type, dp,
                                             collision_term0level, collision_termilevel,
                                             px, py, pz, p0,
                                             npx, npy, npz, nx, ny, nz, i_level,
                                             num_momentum_level, i_grid,
                                             middle_npx, middle_npy, middle_npz, 
                                             hbar, c, lambdax)




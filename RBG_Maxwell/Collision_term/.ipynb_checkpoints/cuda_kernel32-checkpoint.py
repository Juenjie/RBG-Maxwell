from numba import float64
import math
from numba import cuda        
from numba.cuda.random import xoroshiro128p_uniform_float64
from .. import Collision_database

# import the correct amplitude
with open(Collision_database.__path__[0]+'/selected_system.txt','r') as save_text:
    string = 'from ..Collision_database.selected_system.amplitude_square32 import Amplitude_square'.replace("selected_system",save_text.read()) 
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
def accumulate_collision_term(f_x_p_t0level, f_x_p_tilevel, 
                              m1_squared, k1x, k1y, k1z, m2_squared, 
                              m3_squared, k3x, k3y, k3z, k30, 
                              m4_squared, k4x, k4y, k4z, k40,
                              mp_squared, px, py, pz, p0,
                              half_px, half_py, half_pz, dp, masses,
                              ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                              p_type, i_grid, particle_type, degeneracy,
                              flavor, collision_type, i_collision, 
                              collision_term0level, collision_termilevel,
                              npx, npy, npz, nx, ny, nz, 
                              i_level, num_momentum_level, 
                              middle_npx, middle_npy, middle_npz,
                              int_volume, hbar, c, lambdax):
    
    # k1z must be in the momentum box range of particle type 0 
    if k1z > -half_pz[flavor[i_collision,0]] and k1z < half_pz[flavor[i_collision,0]]:
        
        # obtain k2x, k2y, k2z via momentum conservation
        k2x = k3x + k4x + px - k1x
        k2y = k3y + k4y + py - k1y
        k2z = k3z + k4z + pz - k1z
        
        # k2x, k2y, k2z must also be in the momentum box range of particle 1
        if (k2x > -half_px[flavor[i_collision,1]] and k2x < half_px[flavor[i_collision,1]] and 
            k2y > -half_py[flavor[i_collision,1]] and k2y < half_py[flavor[i_collision,1]] and 
            k2z > -half_pz[flavor[i_collision,1]] and k2z < half_pz[flavor[i_collision,1]]):
            
            # energy for particle 1 and 2
            k10, k20 = math.sqrt(m1_squared*c**2+k1x**2+k1y**2+k1z**2), math.sqrt(m2_squared*c**2+k2x**2+k2y**2+k2z**2)

            # jacobin term
            Ja = abs(k1z/k10 - (-k1z-k3z+k4z+pz)/k20)
            # momentum product
            momentum_product = k10*k20*k30*k40*p0
                
            # momentum_producted cannot be zero
            # Ja should be around 1, hence we exclude Ja that are 5 magnitudes smaller
            if Ja>1e-5 and momentum_product > 10**(-19):
                
                ################################################################################
                ################################################################################
                # for particle 0
                ik1x, ik1y, ik1z = int((k1x + half_px[flavor[i_collision,0]])//dpx[flavor[i_collision,0]]), \
                                   int((k1y + half_py[flavor[i_collision,0]])//dpy[flavor[i_collision,0]]), \
                                   int((k1z + half_pz[flavor[i_collision,0]])//dpz[flavor[i_collision,0]])
                # convert grid index in one dimension
                i_phase_grid0 = threeD_to_oneD(ix,iy,iz,ik1x,ik1y,ik1z, \
                                               nx, ny, nz, npx, npy, npz)
    
                ################################################################################ 
                ################################################################################
                # for particle 1
                ik2x, ik2y, ik2z = int((k2x + half_px[flavor[i_collision,1]])//dpx[flavor[i_collision,1]]), \
                                   int((k2y + half_py[flavor[i_collision,1]])//dpy[flavor[i_collision,1]]), \
                                   int((k2z + half_pz[flavor[i_collision,1]])//dpz[flavor[i_collision,1]])
                # convert grid index in one dimension
                i_phase_grid1 = threeD_to_oneD(ix,iy,iz,ik2x,ik2y,ik2z, \
                                               nx, ny, nz, npx, npy, npz)

                ################################################################################
                ################################################################################
                # for particle 2
                ik3x, ik3y, ik3z = int((k3x + half_px[flavor[i_collision,2]])//dpx[flavor[i_collision,2]]), \
                                   int((k3y + half_py[flavor[i_collision,2]])//dpy[flavor[i_collision,2]]), \
                                   int((k3z + half_pz[flavor[i_collision,2]])//dpz[flavor[i_collision,2]])
                # convert grid index in one dimension
                i_phase_grid2 = threeD_to_oneD(ix,iy,iz,ik3x,ik3y,ik3z, \
                                               nx, ny, nz, npx, npy, npz)
      
                ################################################################################
                ################################################################################
                # for particle 3
                ik4x, ik4y, ik4z = int((k4x + half_px[flavor[i_collision,3]])//dpx[flavor[i_collision,3]]), \
                                   int((k4y + half_py[flavor[i_collision,3]])//dpy[flavor[i_collision,3]]), \
                                   int((k4z + half_pz[flavor[i_collision,3]])//dpz[flavor[i_collision,3]])
                # convert grid index in one dimension
                i_phase_grid3 = threeD_to_oneD(ix,iy,iz,ik4x,ik4y,ik4z, \
                                               nx, ny, nz, npx, npy, npz)
      
                ################################################################################
                ################################################################################
                    
                # distribution function: f1,f2,f3,f4,fp
                f = cuda.local.array(shape=5, dtype=float64)
                f[0] = f_x_p_t0level[flavor[i_collision,0], i_phase_grid0]
                f[1] = f_x_p_t0level[flavor[i_collision,1], i_phase_grid1]
                f[2] = f_x_p_t0level[flavor[i_collision,2], i_phase_grid2]
                f[3] = f_x_p_t0level[flavor[i_collision,3], i_phase_grid3]
                f[4] = f_x_p_tilevel[p_type, i_grid]
                
                # feed values of distribution function and its quantum correction
                tf = cuda.local.array(shape=5, dtype=float64)
                # particle_type: 0, 1, 2 for classical, fermi and Bosonic
                for i_particle in range(5):
                    i_flavor = flavor[i_collision,i_particle]
                    if particle_type[i_flavor] == 1:
                        tf[i_particle] = 1 - (2*math.pi*hbar)**3*f[i_particle]/degeneracy[i_flavor]
                    elif particle_type[i_flavor] == 2:
                        tf[i_particle] = 1 + (2*math.pi*hbar)**3*f[i_particle]/degeneracy[i_flavor]
                    else:
                        tf[i_particle] = 1
                        
                # f1*f2*tf3*tf4*tfp - tf1*tf2*f3*f4*fp
                distribution_terms = f[3]*f[4]*tf[0]*tf[1]*tf[2]/(2*math.pi*hbar)**3\
                                     - tf[3]*tf[4]*f[0]*f[1]*f[2]
                
                # amplitude square
                amplitude_square = Amplitude_square(m1_squared,m2_squared,m3_squared,m4_squared,mp_squared,\
                                                    k10,k20,k30,k40,p0,collision_type[i_collision],\
                                                    k1x,k1y,k1z,k2x,k2y,k2z,k3x,k3y,k3z,k4x,k4y,k4z,\
                                                    px,py,pz,hbar,c,lambdax)

                # symmetry factor = 0.5 for same incoming particle species 
                # and 1. for differnent particle species
                if flavor[i_collision,0]==flavor[i_collision,1]==flavor[i_collision,2]:
                    symmetry_factor = 1/6
                elif flavor[i_collision,0]==flavor[i_collision,1] or \
                     flavor[i_collision,1]==flavor[i_collision,2] or \
                     flavor[i_collision,0]==flavor[i_collision,2]:
                    symmetry_factor = 0.5
                else:
                    symmetry_factor = 1.0
                
                # accumulate collision kernel
                # some factors are compensated later
                result = distribution_terms*amplitude_square*symmetry_factor*\
                         hbar**3*c*int_volume/(momentum_product*128*math.pi**2)/Ja

                cuda.atomic.add(collision_term0level, \
                            (flavor[i_collision,0], i_phase_grid0), -result/dp[flavor[i_collision,0]]*dp[p_type])
                cuda.atomic.add(collision_term0level, \
                            (flavor[i_collision,1], i_phase_grid1), -result/dp[flavor[i_collision,1]]*dp[p_type])
                cuda.atomic.add(collision_term0level, \
                            (flavor[i_collision,2], i_phase_grid2), result/dp[flavor[i_collision,2]]*dp[p_type])   
                cuda.atomic.add(collision_term0level, \
                            (flavor[i_collision,3], i_phase_grid2), result/dp[flavor[i_collision,3]]*dp[p_type])
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
def H_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,m4,k40,k4x,k4y,k4z,p0,px,py,pz,c):
    return (-4*k30**2*k3z - 8*k1x*k3x*k3z + \
4*k3x**2*k3z - 8*k1y*k3y*k3z + \
4*k3y**2*k3z + 4*k3z**3 - \
8*k30*k3z*k40 - 4*k3z*k40**2 - \
8*k1x*k3z*k4x + 8*k3x*k3z*k4x + \
4*k3z*k4x**2 - 8*k1y*k3z*k4y + \
8*k3y*k3z*k4y + 4*k3z*k4y**2 - \
4*k30**2*k4z - 8*k1x*k3x*k4z + \
4*k3x**2*k4z - 8*k1y*k3y*k4z + \
4*k3y**2*k4z + 12*k3z**2*k4z - \
8*k30*k40*k4z - 4*k40**2*k4z - \
8*k1x*k4x*k4z + 8*k3x*k4x*k4z + \
4*k4x**2*k4z - 8*k1y*k4y*k4z + \
8*k3y*k4y*k4z + 4*k4y**2*k4z + \
12*k3z*k4z**2 + 4*k4z**3 - \
4*c**2*k3z*m1**2 - \
4*c**2*k4z*m1**2 + \
4*c**2*k3z*m2**2 + \
4*c**2*k4z*m2**2 - 8*k30*k3z*p0 - \
8*k3z*k40*p0 - 8*k30*k4z*p0 - \
8*k40*k4z*p0 - 4*k3z*p0**2 - \
4*k4z*p0**2 - 8*k1x*k3z*px + \
8*k3x*k3z*px + 8*k3z*k4x*px - \
8*k1x*k4z*px + 8*k3x*k4z*px + \
8*k4x*k4z*px + 4*k3z*px**2 + \
4*k4z*px**2 - 8*k1y*k3z*py + \
8*k3y*k3z*py + 8*k3z*k4y*py - \
8*k1y*k4z*py + 8*k3y*k4z*py + \
8*k4y*k4z*py + 4*k3z*py**2 + \
4*k4z*py**2 - 4*k30**2*pz - \
8*k1x*k3x*pz + 4*k3x**2*pz - \
8*k1y*k3y*pz + 4*k3y**2*pz + \
12*k3z**2*pz - 8*k30*k40*pz - \
4*k40**2*pz - 8*k1x*k4x*pz + \
8*k3x*k4x*pz + 4*k4x**2*pz - \
8*k1y*k4y*pz + 8*k3y*k4y*pz + \
4*k4y**2*pz + 24*k3z*k4z*pz + \
12*k4z**2*pz - 4*c**2*m1**2*pz + \
4*c**2*m2**2*pz - 8*k30*p0*pz - \
8*k40*p0*pz - 4*p0**2*pz - \
8*k1x*px*pz + 8*k3x*px*pz + \
8*k4x*px*pz + 4*px**2*pz - \
8*k1y*py*pz + 8*k3y*py*pz + \
8*k4y*py*pz + 4*py**2*pz + \
12*k3z*pz**2 + 12*k4z*pz**2 + \
4*pz**3)**2 - \
4*(4*k30**2 - 4*k3z**2 + 8*k30*k40 + \
4*k40**2 - 8*k3z*k4z - 4*k4z**2 + \
8*k30*p0 + 8*k40*p0 + 4*p0**2 - \
8*k3z*pz - 8*k4z*pz - 4*pz**2)*\
(4*k1x**2*k30**2 + 4*k1y**2*k30**2 - \
k30**4 - 4*k1x*k30**2*k3x - \
4*k1x**2*k3x**2 + 2*k30**2*k3x**2 + \
4*k1x*k3x**3 - k3x**4 - \
4*k1y*k30**2*k3y - \
8*k1x*k1y*k3x*k3y + \
4*k1y*k3x**2*k3y - \
4*k1y**2*k3y**2 + 2*k30**2*k3y**2 + \
4*k1x*k3x*k3y**2 - \
2*k3x**2*k3y**2 + 4*k1y*k3y**3 - \
k3y**4 + 2*k30**2*k3z**2 + \
4*k1x*k3x*k3z**2 - \
2*k3x**2*k3z**2 + \
4*k1y*k3y*k3z**2 - \
2*k3y**2*k3z**2 - k3z**4 + \
8*k1x**2*k30*k40 + \
8*k1y**2*k30*k40 - 4*k30**3*k40 - \
8*k1x*k30*k3x*k40 + \
4*k30*k3x**2*k40 - \
8*k1y*k30*k3y*k40 + \
4*k30*k3y**2*k40 + \
4*k30*k3z**2*k40 + \
4*k1x**2*k40**2 + 4*k1y**2*k40**2 - \
6*k30**2*k40**2 - \
4*k1x*k3x*k40**2 + \
2*k3x**2*k40**2 - \
4*k1y*k3y*k40**2 + \
2*k3y**2*k40**2 + 2*k3z**2*k40**2 - \
4*k30*k40**3 - k40**4 - \
4*k1x*k30**2*k4x - \
8*k1x**2*k3x*k4x + \
4*k30**2*k3x*k4x + \
12*k1x*k3x**2*k4x - 4*k3x**3*k4x - \
8*k1x*k1y*k3y*k4x + \
8*k1y*k3x*k3y*k4x + \
4*k1x*k3y**2*k4x - \
4*k3x*k3y**2*k4x + \
4*k1x*k3z**2*k4x - \
4*k3x*k3z**2*k4x - \
8*k1x*k30*k40*k4x + \
8*k30*k3x*k40*k4x - \
4*k1x*k40**2*k4x + \
4*k3x*k40**2*k4x - \
4*k1x**2*k4x**2 + 2*k30**2*k4x**2 + \
12*k1x*k3x*k4x**2 - \
6*k3x**2*k4x**2 + \
4*k1y*k3y*k4x**2 - \
2*k3y**2*k4x**2 - 2*k3z**2*k4x**2 + \
4*k30*k40*k4x**2 + \
2*k40**2*k4x**2 + 4*k1x*k4x**3 - \
4*k3x*k4x**3 - k4x**4 - \
4*k1y*k30**2*k4y - \
8*k1x*k1y*k3x*k4y + \
4*k1y*k3x**2*k4y - \
8*k1y**2*k3y*k4y + \
4*k30**2*k3y*k4y + \
8*k1x*k3x*k3y*k4y - \
4*k3x**2*k3y*k4y + \
12*k1y*k3y**2*k4y - 4*k3y**3*k4y + \
4*k1y*k3z**2*k4y - \
4*k3y*k3z**2*k4y - \
8*k1y*k30*k40*k4y + \
8*k30*k3y*k40*k4y - \
4*k1y*k40**2*k4y + \
4*k3y*k40**2*k4y - \
8*k1x*k1y*k4x*k4y + \
8*k1y*k3x*k4x*k4y + \
8*k1x*k3y*k4x*k4y - \
8*k3x*k3y*k4x*k4y + \
4*k1y*k4x**2*k4y - \
4*k3y*k4x**2*k4y - \
4*k1y**2*k4y**2 + 2*k30**2*k4y**2 + \
4*k1x*k3x*k4y**2 - \
2*k3x**2*k4y**2 + \
12*k1y*k3y*k4y**2 - \
6*k3y**2*k4y**2 - 2*k3z**2*k4y**2 + \
4*k30*k40*k4y**2 + \
2*k40**2*k4y**2 + \
4*k1x*k4x*k4y**2 - \
4*k3x*k4x*k4y**2 - \
2*k4x**2*k4y**2 + 4*k1y*k4y**3 - 
4*k3y*k4y**3 - k4y**4 + 
4*k30**2*k3z*k4z + 
8*k1x*k3x*k3z*k4z - 
4*k3x**2*k3z*k4z + 
8*k1y*k3y*k3z*k4z - 
4*k3y**2*k3z*k4z - 4*k3z**3*k4z + 
8*k30*k3z*k40*k4z + 
4*k3z*k40**2*k4z + 
8*k1x*k3z*k4x*k4z - 
8*k3x*k3z*k4x*k4z - 
4*k3z*k4x**2*k4z + 
8*k1y*k3z*k4y*k4z - 
8*k3y*k3z*k4y*k4z - 
4*k3z*k4y**2*k4z + 
2*k30**2*k4z**2 + 
4*k1x*k3x*k4z**2 - 
2*k3x**2*k4z**2 + 
4*k1y*k3y*k4z**2 - 
2*k3y**2*k4z**2 - 6*k3z**2*k4z**2 + 
4*k30*k40*k4z**2 + 
2*k40**2*k4z**2 + 
4*k1x*k4x*k4z**2 - 
4*k3x*k4x*k4z**2 - 
2*k4x**2*k4z**2 + 
4*k1y*k4y*k4z**2 - 
4*k3y*k4y*k4z**2 - 
2*k4y**2*k4z**2 - 4*k3z*k4z**3 - 
k4z**4 + 2*c**2*k30**2*m1**2 - 
4*c**2*k1x*k3x*m1**2 + 
2*c**2*k3x**2*m1**2 - 
4*c**2*k1y*k3y*m1**2 + 
2*c**2*k3y**2*m1**2 + 
2*c**2*k3z**2*m1**2 + 
4*c**2*k30*k40*m1**2 + 
2*c**2*k40**2*m1**2 - 
4*c**2*k1x*k4x*m1**2 + 
4*c**2*k3x*k4x*m1**2 + 
2*c**2*k4x**2*m1**2 - 
4*c**2*k1y*k4y*m1**2 + 
4*c**2*k3y*k4y*m1**2 + 
2*c**2*k4y**2*m1**2 + 
4*c**2*k3z*k4z*m1**2 + 
2*c**2*k4z**2*m1**2 - c**4*m1**4 + 
2*c**2*k30**2*m2**2 + 
4*c**2*k1x*k3x*m2**2 - 
2*c**2*k3x**2*m2**2 + 
4*c**2*k1y*k3y*m2**2 - 
2*c**2*k3y**2*m2**2 - 
2*c**2*k3z**2*m2**2 + 
4*c**2*k30*k40*m2**2 + 
2*c**2*k40**2*m2**2 + 
4*c**2*k1x*k4x*m2**2 - 
4*c**2*k3x*k4x*m2**2 - 
2*c**2*k4x**2*m2**2 + 
4*c**2*k1y*k4y*m2**2 - 
4*c**2*k3y*k4y*m2**2 - 
2*c**2*k4y**2*m2**2 - 
4*c**2*k3z*k4z*m2**2 - 
2*c**2*k4z**2*m2**2 + 
2*c**4*m1**2*m2**2 - c**4*m2**4 + 
8*k1x**2*k30*p0 + 8*k1y**2*k30*p0 - 
4*k30**3*p0 - 8*k1x*k30*k3x*p0 + 
4*k30*k3x**2*p0 - 
8*k1y*k30*k3y*p0 + 
4*k30*k3y**2*p0 + 4*k30*k3z**2*p0 + 
8*k1x**2*k40*p0 + 8*k1y**2*k40*p0 - 
12*k30**2*k40*p0 - 
8*k1x*k3x*k40*p0 + 
4*k3x**2*k40*p0 - 
8*k1y*k3y*k40*p0 + 
4*k3y**2*k40*p0 + 4*k3z**2*k40*p0 - 
12*k30*k40**2*p0 - 4*k40**3*p0 - 
8*k1x*k30*k4x*p0 + 
8*k30*k3x*k4x*p0 - 
8*k1x*k40*k4x*p0 + 
8*k3x*k40*k4x*p0 + 
4*k30*k4x**2*p0 + 4*k40*k4x**2*p0 - 
8*k1y*k30*k4y*p0 + 
8*k30*k3y*k4y*p0 - 
8*k1y*k40*k4y*p0 + 
8*k3y*k40*k4y*p0 + 
4*k30*k4y**2*p0 + 4*k40*k4y**2*p0 + 
8*k30*k3z*k4z*p0 + 
8*k3z*k40*k4z*p0 + 
4*k30*k4z**2*p0 + 4*k40*k4z**2*p0 + 
4*c**2*k30*m1**2*p0 + 
4*c**2*k40*m1**2*p0 + 
4*c**2*k30*m2**2*p0 + 
4*c**2*k40*m2**2*p0 + 
4*k1x**2*p0**2 + 4*k1y**2*p0**2 - 
6*k30**2*p0**2 - 4*k1x*k3x*p0**2 + 
2*k3x**2*p0**2 - 4*k1y*k3y*p0**2 + 
2*k3y**2*p0**2 + 2*k3z**2*p0**2 - 
12*k30*k40*p0**2 - 6*k40**2*p0**2 - 
4*k1x*k4x*p0**2 + 4*k3x*k4x*p0**2 + 
2*k4x**2*p0**2 - 4*k1y*k4y*p0**2 + 
4*k3y*k4y*p0**2 + 2*k4y**2*p0**2 + 
4*k3z*k4z*p0**2 + 2*k4z**2*p0**2 + 
2*c**2*m1**2*p0**2 + 
2*c**2*m2**2*p0**2 - 4*k30*p0**3 - 
4*k40*p0**3 - p0**4 - 
4*k1x*k30**2*px - 8*k1x**2*k3x*px + 
4*k30**2*k3x*px + 
12*k1x*k3x**2*px - 4*k3x**3*px - 
8*k1x*k1y*k3y*px + 
8*k1y*k3x*k3y*px + 
4*k1x*k3y**2*px - 4*k3x*k3y**2*px + 
4*k1x*k3z**2*px - 4*k3x*k3z**2*px - 
8*k1x*k30*k40*px + 
8*k30*k3x*k40*px - 
4*k1x*k40**2*px + 4*k3x*k40**2*px - 
8*k1x**2*k4x*px + 4*k30**2*k4x*px + 
24*k1x*k3x*k4x*px - 
12*k3x**2*k4x*px + 
8*k1y*k3y*k4x*px - 
4*k3y**2*k4x*px - 4*k3z**2*k4x*px + 
8*k30*k40*k4x*px + 
4*k40**2*k4x*px + 
12*k1x*k4x**2*px - 
12*k3x*k4x**2*px - 4*k4x**3*px - 
8*k1x*k1y*k4y*px + 
8*k1y*k3x*k4y*px + 
8*k1x*k3y*k4y*px - 
8*k3x*k3y*k4y*px + 
8*k1y*k4x*k4y*px - 
8*k3y*k4x*k4y*px + 
4*k1x*k4y**2*px - 4*k3x*k4y**2*px - 
4*k4x*k4y**2*px + 
8*k1x*k3z*k4z*px - 
8*k3x*k3z*k4z*px - 
8*k3z*k4x*k4z*px + 
4*k1x*k4z**2*px - 4*k3x*k4z**2*px - 
4*k4x*k4z**2*px - 
4*c**2*k1x*m1**2*px + 
4*c**2*k3x*m1**2*px + 
4*c**2*k4x*m1**2*px + 
4*c**2*k1x*m2**2*px - 
4*c**2*k3x*m2**2*px - 
4*c**2*k4x*m2**2*px - 
8*k1x*k30*p0*px + 8*k30*k3x*p0*px - 
8*k1x*k40*p0*px + 8*k3x*k40*p0*px + 
8*k30*k4x*p0*px + 8*k40*k4x*p0*px - 
4*k1x*p0**2*px + 4*k3x*p0**2*px + 
4*k4x*p0**2*px - 4*k1x**2*px**2 + 
2*k30**2*px**2 + 12*k1x*k3x*px**2 - 
6*k3x**2*px**2 + 4*k1y*k3y*px**2 - 
2*k3y**2*px**2 - 2*k3z**2*px**2 + 
4*k30*k40*px**2 + 2*k40**2*px**2 + 
12*k1x*k4x*px**2 - 
12*k3x*k4x*px**2 - 6*k4x**2*px**2 + 
4*k1y*k4y*px**2 - 4*k3y*k4y*px**2 - 
2*k4y**2*px**2 - 4*k3z*k4z*px**2 - 
2*k4z**2*px**2 + 
2*c**2*m1**2*px**2 - 
2*c**2*m2**2*px**2 + 
4*k30*p0*px**2 + 4*k40*p0*px**2 + 
2*p0**2*px**2 + 4*k1x*px**3 - 
4*k3x*px**3 - 4*k4x*px**3 - px**4 - 
4*k1y*k30**2*py - 
8*k1x*k1y*k3x*py + 
4*k1y*k3x**2*py - 8*k1y**2*k3y*py + 
4*k30**2*k3y*py + 
8*k1x*k3x*k3y*py - 
4*k3x**2*k3y*py + 
12*k1y*k3y**2*py - 4*k3y**3*py + 
4*k1y*k3z**2*py - 4*k3y*k3z**2*py - 
8*k1y*k30*k40*py + 
8*k30*k3y*k40*py - 
4*k1y*k40**2*py + 4*k3y*k40**2*py - 
8*k1x*k1y*k4x*py + 
8*k1y*k3x*k4x*py + 
8*k1x*k3y*k4x*py - 
8*k3x*k3y*k4x*py + 
4*k1y*k4x**2*py - 4*k3y*k4x**2*py - 
8*k1y**2*k4y*py + 4*k30**2*k4y*py + 
8*k1x*k3x*k4y*py - 
4*k3x**2*k4y*py + 
24*k1y*k3y*k4y*py - 
12*k3y**2*k4y*py - 
4*k3z**2*k4y*py + 
8*k30*k40*k4y*py + 
4*k40**2*k4y*py + 
8*k1x*k4x*k4y*py - 
8*k3x*k4x*k4y*py - 
4*k4x**2*k4y*py + 
12*k1y*k4y**2*py - 
12*k3y*k4y**2*py - 4*k4y**3*py + 
8*k1y*k3z*k4z*py - 
8*k3y*k3z*k4z*py - 
8*k3z*k4y*k4z*py + 
4*k1y*k4z**2*py - 4*k3y*k4z**2*py - 
4*k4y*k4z**2*py - 
4*c**2*k1y*m1**2*py + 
4*c**2*k3y*m1**2*py + 
4*c**2*k4y*m1**2*py + 
4*c**2*k1y*m2**2*py - 
4*c**2*k3y*m2**2*py - 
4*c**2*k4y*m2**2*py - 
8*k1y*k30*p0*py + 8*k30*k3y*p0*py - 
8*k1y*k40*p0*py + 8*k3y*k40*p0*py + 
8*k30*k4y*p0*py + 8*k40*k4y*p0*py - 
4*k1y*p0**2*py + 4*k3y*p0**2*py + 
4*k4y*p0**2*py - 8*k1x*k1y*px*py + 
8*k1y*k3x*px*py + 8*k1x*k3y*px*py - 
8*k3x*k3y*px*py + 8*k1y*k4x*px*py - 
8*k3y*k4x*px*py + 8*k1x*k4y*px*py - 
8*k3x*k4y*px*py - 8*k4x*k4y*px*py + 
4*k1y*px**2*py - 4*k3y*px**2*py - 
4*k4y*px**2*py - 4*k1y**2*py**2 + 
2*k30**2*py**2 + 4*k1x*k3x*py**2 - 
2*k3x**2*py**2 + 12*k1y*k3y*py**2 - 
6*k3y**2*py**2 - 2*k3z**2*py**2 + 
4*k30*k40*py**2 + 2*k40**2*py**2 + 
4*k1x*k4x*py**2 - 4*k3x*k4x*py**2 - 
2*k4x**2*py**2 + 12*k1y*k4y*py**2 - 
12*k3y*k4y*py**2 - 6*k4y**2*py**2 - 
4*k3z*k4z*py**2 - 2*k4z**2*py**2 + 
2*c**2*m1**2*py**2 - 
2*c**2*m2**2*py**2 + 
4*k30*p0*py**2 + 4*k40*p0*py**2 + 
2*p0**2*py**2 + 4*k1x*px*py**2 - 
4*k3x*px*py**2 - 4*k4x*px*py**2 - 
2*px**2*py**2 + 4*k1y*py**3 - 
4*k3y*py**3 - 4*k4y*py**3 - py**4 + 
4*k30**2*k3z*pz + 
8*k1x*k3x*k3z*pz - 
4*k3x**2*k3z*pz + 
8*k1y*k3y*k3z*pz - 
4*k3y**2*k3z*pz - 4*k3z**3*pz + 
8*k30*k3z*k40*pz + 
4*k3z*k40**2*pz + 
8*k1x*k3z*k4x*pz - 
8*k3x*k3z*k4x*pz - 
4*k3z*k4x**2*pz + 
8*k1y*k3z*k4y*pz - 
8*k3y*k3z*k4y*pz - 
4*k3z*k4y**2*pz + 4*k30**2*k4z*pz + 
8*k1x*k3x*k4z*pz - 
4*k3x**2*k4z*pz + 
8*k1y*k3y*k4z*pz - 
4*k3y**2*k4z*pz - 
12*k3z**2*k4z*pz + 
8*k30*k40*k4z*pz + 
4*k40**2*k4z*pz + 
8*k1x*k4x*k4z*pz - 
8*k3x*k4x*k4z*pz - 
4*k4x**2*k4z*pz + 
8*k1y*k4y*k4z*pz - 
8*k3y*k4y*k4z*pz - 
4*k4y**2*k4z*pz - 
12*k3z*k4z**2*pz - 4*k4z**3*pz + 
4*c**2*k3z*m1**2*pz + 
4*c**2*k4z*m1**2*pz - 
4*c**2*k3z*m2**2*pz - 
4*c**2*k4z*m2**2*pz + 
8*k30*k3z*p0*pz + 8*k3z*k40*p0*pz + 
8*k30*k4z*p0*pz + 8*k40*k4z*p0*pz + 
4*k3z*p0**2*pz + 4*k4z*p0**2*pz + 
8*k1x*k3z*px*pz - 8*k3x*k3z*px*pz - 
8*k3z*k4x*px*pz + 8*k1x*k4z*px*pz - 
8*k3x*k4z*px*pz - 8*k4x*k4z*px*pz - 
4*k3z*px**2*pz - 4*k4z*px**2*pz + 
8*k1y*k3z*py*pz - 8*k3y*k3z*py*pz - 
8*k3z*k4y*py*pz + 8*k1y*k4z*py*pz - 
8*k3y*k4z*py*pz - 8*k4y*k4z*py*pz - 
4*k3z*py**2*pz - 4*k4z*py**2*pz + 
2*k30**2*pz**2 + 4*k1x*k3x*pz**2 - 
2*k3x**2*pz**2 + 4*k1y*k3y*pz**2 - 
2*k3y**2*pz**2 - 6*k3z**2*pz**2 + 
4*k30*k40*pz**2 + 2*k40**2*pz**2 + 
4*k1x*k4x*pz**2 - 4*k3x*k4x*pz**2 - 
2*k4x**2*pz**2 + 4*k1y*k4y*pz**2 - 
4*k3y*k4y*pz**2 - 2*k4y**2*pz**2 - 
12*k3z*k4z*pz**2 - 6*k4z**2*pz**2 + 
2*c**2*m1**2*pz**2 - 
2*c**2*m2**2*pz**2 + 
4*k30*p0*pz**2 + 4*k40*p0*pz**2 + 
2*p0**2*pz**2 + 4*k1x*px*pz**2 - 
4*k3x*px*pz**2 - 4*k4x*px*pz**2 - 
2*px**2*pz**2 + 4*k1y*py*pz**2 - 
4*k3y*py*pz**2 - 4*k4y*py*pz**2 - 
2*py**2*pz**2 - 4*k3z*pz**3 - 
4*k4z*pz**3 - pz**4)

@cuda.jit(device=True)
def C1_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,m4,k40,k4x,k4y,k4z,p0,px,py,pz,c):
    return 4*k30**2*k3z + 8*k1x*k3x*k3z - \
4*k3x**2*k3z + 8*k1y*k3y*k3z - \
4*k3y**2*k3z - 4*k3z**3 + \
8*k30*k3z*k40 + 4*k3z*k40**2 + \
8*k1x*k3z*k4x - 8*k3x*k3z*k4x - \
4*k3z*k4x**2 + 8*k1y*k3z*k4y - \
8*k3y*k3z*k4y - 4*k3z*k4y**2 + \
4*k30**2*k4z + 8*k1x*k3x*k4z - \
4*k3x**2*k4z + 8*k1y*k3y*k4z - \
4*k3y**2*k4z - 12*k3z**2*k4z + \
8*k30*k40*k4z + 4*k40**2*k4z + \
8*k1x*k4x*k4z - 8*k3x*k4x*k4z - \
4*k4x**2*k4z + 8*k1y*k4y*k4z - \
8*k3y*k4y*k4z - 4*k4y**2*k4z - \
12*k3z*k4z**2 - 4*k4z**3 + \
4*c**2*k3z*m1**2 + 4*c**2*k4z*m1**2 - \
4*c**2*k3z*m2**2 - 4*c**2*k4z*m2**2 + \
8*k30*k3z*p0 + 8*k3z*k40*p0 + \
8*k30*k4z*p0 + 8*k40*k4z*p0 + \
4*k3z*p0**2 + 4*k4z*p0**2 + \
8*k1x*k3z*px - 8*k3x*k3z*px - \
8*k3z*k4x*px + 8*k1x*k4z*px - \
8*k3x*k4z*px - 8*k4x*k4z*px - \
4*k3z*px**2 - 4*k4z*px**2 + \
8*k1y*k3z*py - 8*k3y*k3z*py - \
8*k3z*k4y*py + 8*k1y*k4z*py - \
8*k3y*k4z*py - 8*k4y*k4z*py - \
4*k3z*py**2 - 4*k4z*py**2 + \
4*k30**2*pz + 8*k1x*k3x*pz - \
4*k3x**2*pz + 8*k1y*k3y*pz - \
4*k3y**2*pz - 12*k3z**2*pz + \
8*k30*k40*pz + 4*k40**2*pz + \
8*k1x*k4x*pz - 8*k3x*k4x*pz - \
4*k4x**2*pz + 8*k1y*k4y*pz - \
8*k3y*k4y*pz - 4*k4y**2*pz - \
24*k3z*k4z*pz - 12*k4z**2*pz + \
4*c**2*m1**2*pz - 4*c**2*m2**2*pz + \
8*k30*p0*pz + 8*k40*p0*pz + \
4*p0**2*pz + 8*k1x*px*pz - \
8*k3x*px*pz - 8*k4x*px*pz - \
4*px**2*pz + 8*k1y*py*pz - \
8*k3y*py*pz - 8*k4y*py*pz - \
4*py**2*pz - 12*k3z*pz**2 - \
12*k4z*pz**2 - 4*pz**3

@cuda.jit(device=True)
def C2_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,m4,k40,k4x,k4y,k4z,p0,px,py,pz,c):
    return 2*(4*k30**2 - 4*k3z**2 + 8*k30*k40 + 
4*k40**2 - 8*k3z*k4z - 4*k4z**2 + 
8*k30*p0 + 8*k40*p0 + 4*p0**2 - 
8*k3z*pz - 8*k4z*pz - 4*pz**2)

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
    H+ (0), H2 (1), H (2), H2+ (3), e- (4).

    for 2-2 collisions:
        (type 0): H+ + H2 <-> H  + H2+
        (type 1): H+ + H2 <-> H+ + H2
        (type 2): H2 + e- <-> e- + H2

    for 2-3 collisions:
        (type 0): H+ + H2 <-> H+ + H2+ + e-
        (type 1): H  + H2 <-> H+ + H2  + e-
        (type 2): H  + H2 <-> H  + H2+ + e-
        (type 3): e- + H2 <-> e- + H2+ + e-
    # indicating final H+, H2, H, H2+, e-
    # use 10001 to occupy empty space
    collision_type22=np.array([[1,10001],[1,2],[0,10001],[0,10001],[2,10001]],dtype=np.int64)

    flavor22=np.array([[[1,0,1,0],    [10001,10001,10001,10001]],
                       [[0,1,0,1],    [4,1,4,1]],
                       [[0,1,3,2],    [10001,10001,10001,10001]],
                       [[0,1,2,3],    [10001,10001,10001,10001]],
                       [[1,4,1,4],    [10001,10001,10001,10001]]],dtype=np.int64)

    # indicating final H+, H2, H, H2+, e-
    # use 10001 to occupy empty space
    collision_type23=np.array([[0,1,10001,10001],\
                               [1,10001,10001,10001],\
                               [2,10001,10001,10001],\
                               [0,2,3,10001],\
                               [0,1,2,3]],dtype=np.int64)

    flavor23=np.array([[[0,1,3,4,0], [1,2,1,4,0],               [10001,10001,10001,10001], [10001,10001,10001,10001]],
                       [[1,2,0,4,1], [10001,10001,10001,10001], [10001,10001,10001,10001], [10001,10001,10001,10001]],
                       [[2,1,3,4,2], [10001,10001,10001,10001], [10001,10001,10001,10001], [10001,10001,10001,10001]],
                       [[0,1,0,4,3], [1,2,2,4,3],               [1,4,4,4,3],               [10001,10001,10001,10001]],
                       [[0,1,0,3,4], [1,2,0,1,4],               [2,1,2,3,4],               [1,4,3,4,4]]],dtype=np.int64)

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
        # 100 indicates that the maximum number of collision types supported is 10000
        if collision_type[i_collision] < 100:
            
            # integration_volume
            # we divide the integration volume by num_samples*5
            # 5 here indicates that the integrations are evaluated for five times due to the symmetric smapling
            # number of samples are for the Monte Carlo integration
            # different particles have different integration domain
            int_volume = 2**8*half_px[flavor[i_collision,0]]*half_py[flavor[i_collision,0]]\
                         *half_px[flavor[i_collision,2]]*half_py[flavor[i_collision,2]]*half_pz[flavor[i_collision,2]]\
                         *half_px[flavor[i_collision,3]]*half_py[flavor[i_collision,3]]*half_pz[flavor[i_collision,3]]\
                         /(num_samples*5)

            # Direct Monte Carlo Method to accumulate point values
            for i_sample in range(num_samples):

                # masses of the particles
                m1 = masses[flavor[i_collision,0]]
                m2 = masses[flavor[i_collision,1]]
                m3 = masses[flavor[i_collision,2]]
                m4 = masses[flavor[i_collision,3]]
                mp = masses[flavor[i_collision,4]]
                
                # randomly generate k1x, k1y, k3x, k3y, k3z
                # the momentum values need to be sampled according to the specific particle types
                k1x = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_px[flavor[i_collision,0]]
                k1y = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_py[flavor[i_collision,0]]
                k3x = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_px[flavor[i_collision,2]]
                k3y = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_py[flavor[i_collision,2]]
                k3z = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_pz[flavor[i_collision,2]]
                k4x = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_px[flavor[i_collision,3]]
                k4y = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_py[flavor[i_collision,3]]
                k4z = (xoroshiro128p_uniform_float64(rng_states, i_grid)*2-1)*half_pz[flavor[i_collision,3]]
                
                k30 = math.sqrt(m3**2*c**2+k3x**2+k3y**2+k3z**2)
                k40 = math.sqrt(m4**2*c**2+k4x**2+k4y**2+k4z**2)

                # solving for k1z via energy conservation
                # k1z has two values
                C1 = C1_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,m4,k40,k4x,k4y,k4z,p0,px,py,pz,c)
                C2 = C2_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,m4,k40,k4x,k4y,k4z,p0,px,py,pz,c)
                H =  H_fun(m1,k1x,k1y,m2,k30,k3x,k3y,k3z,m4,k40,k4x,k4y,k4z,p0,px,py,pz,c)

                # take the value if H>1e-13 and abs(C2)>1e-13
                if H>1e-13 and abs(C2)>1e-13:
                    k1z1, k1z2 = (C1+math.sqrt(H))/C2, (C1-math.sqrt(H))/C2

                    # call accumulate_collision_term for the two values
                    for k1z in (k1z1,k1z2):

                        accumulate_collision_term(f_x_p_t0level, f_x_p_tilevel, 
                                                  m1**2, k1x, k1y, k1z, m2**2, 
                                                  m3**2, k3x, k3y, k3z, k30, 
                                                  m4**2, k4x, k4y, k4z, k40,
                                                  mp_squared, px, py, pz, p0,
                                                  half_px, half_py, half_pz, dp, masses,
                                                  ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                                                  p_type, i_grid, particle_type, degeneracy,
                                                  flavor, collision_type, i_collision, 
                                                  collision_term0level, collision_termilevel,
                                                  npx, npy, npz, nx, ny, nz, 
                                                  i_level, num_momentum_level, 
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
    H+ (0), H2 (1), H (2), H2+ (3), e- (4).

    for 2-2 collisions:
        (type 0): H+ + H2 <-> H  + H2+
        (type 1): H+ + H2 <-> H+ + H2
        (type 2): H2 + e- <-> e- + H2

    for 2-3 collisions:
        (type 0): H+ + H2 <-> H+ + H2+ + e-
        (type 1): H  + H2 <-> H+ + H2  + e-
        (type 2): H  + H2 <-> H  + H2+ + e-
        (type 3): e- + H2 <-> e- + H2+ + e-
    # indicating final H+, H2, H, H2+, e-
    # use 10001 to occupy empty space
    collision_type22=np.array([[1,10001],[1,2],[0,10001],[0,10001],[2,10001]],dtype=np.int64)

    flavor22=np.array([[[1,0,1,0],    [10001,10001,10001,10001]],
                       [[0,1,0,1],    [4,1,4,1]],
                       [[0,1,3,2],    [10001,10001,10001,10001]],
                       [[0,1,2,3],    [10001,10001,10001,10001]],
                       [[1,4,1,4],    [10001,10001,10001,10001]]],dtype=np.int64)

    # indicating final H+, H2, H, H2+, e-
    # use 10001 to occupy empty space
    collision_type23=np.array([[0,1,10001,10001],\
                               [1,10001,10001,10001],\
                               [2,10001,10001,10001],\
                               [0,2,3,10001],\
                               [0,1,2,3]],dtype=np.int64)

    flavor23=np.array([[[0,1,3,4,0], [1,2,1,4,0],               [10001,10001,10001,10001], [10001,10001,10001,10001]],
                       [[1,2,0,4,1], [10001,10001,10001,10001], [10001,10001,10001,10001], [10001,10001,10001,10001]],
                       [[2,1,3,4,2], [10001,10001,10001,10001], [10001,10001,10001,10001], [10001,10001,10001,10001]],
                       [[0,1,0,4,3], [1,2,2,4,3],               [1,4,4,4,3],               [10001,10001,10001,10001]],
                       [[0,1,0,3,4], [1,2,0,1,4],               [2,1,2,3,4],               [1,4,3,4,4]]],dtype=np.int64)
        
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
hbar,c,lambdax:
    numerical value of hbar,c,and lambdax in FU
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





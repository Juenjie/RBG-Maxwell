from numba import cuda
import numpy as np
import math
import cupy
from .cuda_kernel22 import collision_term_kernel as collision_term_kernel22
from .cuda_kernel23 import collision_term_kernel as collision_term_kernel23
from .cuda_kernel32 import collision_term_kernel as collision_term_kernel32

def Collision_term(flavor, collision_type, particle_type, degeneracy,\
                   num_samples, f, masses, total_grid, rng_states, \
                   num_of_particle_types,\
                   npx, npy, npz, nx, ny, nz, dp,\
                   half_px, half_py, half_pz, \
                   dpx, dpy, dpz, blockspergrid_total_phase, threadsperblock, \
                   middle_npx, middle_npy, middle_npz, num_momentum_level, hbar, c, lambdax,\
                   allowed_collosions):
        '''
        Method for obtaining the collision term. This is to be modified by users.

        Params
        ======
        flavor and collision_type:
            dictionary of flavors and collision_types. elements are cupy array.
            
            # the principle for writing the possible collisions is that the collisions
            # must be symmetrical, i.e., if we have a + b -> c + d, we must at the same time
            # consider c + d -> a + b. Note that a + a -> a + a is symmetrical already.
            
            flavor and collision_type:
            dictionary of flavors and collision_types.
            
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
        num_samples = 100:
            number of samples in Monte Carlo integrations
        f:
            distribution function of all particles at the current momentum level
        masses:
            masses for each type of particles, e.g., [0.2, 0.2] GeV
        total_grid:
            total number of grids, nx*ny*nz*npx*npy*npz
        rng_states:
            array for generating random numbers
        num_of_particle_types:
            total number of particle types
        npx,npy,npz: 
            number of grid sizes in in momentum domain, e.g., [5,5,5]
        nx,ny,nz:
            number of grids in spatial domain, e.g., [5, 5, 5]
        dp:
            dpx*dpy*dpz
        half_px, half_py, half_pz:
            left boundary of the momentum region, 
            this is an array containing the values for different type of particles              
        dpx, dpy, dpz:
            infinitesimal difference in momentum coordinate, 
            this is an array containing the values for different type of particles
        blockspergrid_total_phase, threadsperblock:
            GPU kernel ultilization scheme
        middle_npx, middle_npy, middle_npz:
            the middle index of npx, npy, npz
        num_momentum_level:
            total number of momentum levels
        hbar,c,lambdax:
            numerical value of hbar,c,and lambdax in FU
        allowed_collosions:
            list of strings, only support the following three:
            ['2TO2','2TO3','3TO2']
        '''
        
        # initialize collision term
        collision_term = cupy.zeros_like(f)

        # this function is not in use in the current version
        # for i_level in range(num_momentum_level):
        #     # call force kernel for the force term
        #     # momentum quantities will be scaled by the momentum level
        #     collision_term_kernel[blockspergrid_total_phase, threadsperblock](f[0], f[i_level], masses, num_momentum_level,\
        #                                                                       total_grid, num_of_particle_types, \
        #                                                                       npx,npy,npz,nx,ny,nz, \
        #                                                                       half_px, half_py, half_pz, \
        #                                                                       dpx, dpy, dpz, dp,\
        #                                                                       flavor, collision_type, \
        #                                                                       num_samples, rng_states, \
        #                                                                       collision_term[0], collision_term[i_level],\
        #                                                                       middle_npx, middle_npy, middle_npz, i_level,\
        #                                                                       hbar,c,lambdax)
        
        # call force kernel for the force term
        # momentum quantities will be scaled by the momentum level
        num_momentum_level = 1
        if '2TO2' in allowed_collosions:
            collision_term_kernel22[blockspergrid_total_phase, threadsperblock](f[0], f[0], masses, num_momentum_level,\
                                                                              total_grid, num_of_particle_types, \
                                                                              particle_type, degeneracy,\
                                                                              npx,npy,npz,nx,ny,nz, \
                                                                              half_px, half_py, half_pz, \
                                                                              dpx, dpy, dpz, dp,\
                                                                              flavor['2TO2'], collision_type['2TO2'], \
                                                                              num_samples, rng_states, \
                                                                              collision_term[0], collision_term[0],\
                                                                              middle_npx, middle_npy, middle_npz, 0,\
                                                                              hbar,c,lambdax)
            
        if '2TO3' in allowed_collosions:
            collision_term_kernel23[blockspergrid_total_phase, threadsperblock](f[0], f[0], masses, num_momentum_level,\
                                                                              total_grid, num_of_particle_types, \
                                                                              particle_type, degeneracy,\
                                                                              npx,npy,npz,nx,ny,nz, \
                                                                              half_px, half_py, half_pz, \
                                                                              dpx, dpy, dpz, dp,\
                                                                              flavor['2TO3'], collision_type['2TO3'], \
                                                                              num_samples, rng_states, \
                                                                              collision_term[0], collision_term[0],\
                                                                              middle_npx, middle_npy, middle_npz, 0,\
                                                                              hbar,c,lambdax)
            
        if '3TO2' in allowed_collosions:
            collision_term_kernel32[blockspergrid_total_phase, threadsperblock](f[0], f[0], masses, num_momentum_level,\
                                                                              total_grid, num_of_particle_types, \
                                                                              particle_type, degeneracy,\
                                                                              npx,npy,npz,nx,ny,nz, \
                                                                              half_px, half_py, half_pz, \
                                                                              dpx, dpy, dpz, dp,\
                                                                              flavor['3TO2'], collision_type['3TO2'], \
                                                                              num_samples, rng_states, \
                                                                              collision_term[0], collision_term[0],\
                                                                              middle_npx, middle_npy, middle_npz, 0,\
                                                                              hbar,c,lambdax)
        return collision_term
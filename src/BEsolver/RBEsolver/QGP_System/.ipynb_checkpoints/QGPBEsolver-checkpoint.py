import cupy, ray, math, random
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from ..Base.BEsolver import BEsolver
from .collision_term.collision_term_kernel import collision_term_kernel
from .rho_J_kernel import rho_J_kernel

class QGPBEsolver(BEsolver):
    def __init__(self, \
                 dx, dy, dz, x_left_bound, y_left_bound, z_left_bound, \
                 dpx, dpy, dpz, px_left_bound, py_left_bound, pz_left_bound, \
                 x_grid_size, y_grid_size, z_grid_size, \
                 px_grid_size, py_grid_size, pz_grid_size, \
                 initial_distribution, masses, particle_type, charges):
        '''
        Base class for solving the Relativistic Boltzmann equation on GPUs.
        The Unit is Rationalized-Lorentz-Heveaside-QCD

        Params
        ======
        dx, dy, dz: 
            infinitesimal difference in spatial coordinate, these are in GeV^-1 (recomended)
        x_left_bound, y_left_bound, z_left_bound: 
            left boundary of the spatial region, in unit GeV^-1
        dpx, dpy, dpz:
            infinitesimal difference in momentum coordinate, these are in GeV (recomended)
        px_left_bound, py_left_bound, pz_left_bound:
            left boundary of the momentum region, in unit GeV
        x_grid_size, y_grid_size, z_grid_size:
            number of grids in spatial domain, e.g., [5, 5, 5]
        px_grid_size, py_grid_size, pz_grid_size: 
            number of grid sizes in in momentum domain, e.g., [5,5,5]
        initial_distribution:
            initial distribution of the particles, of shape [n_type, nx, ny, nz, npx, npy, npz]
        masses:
            masses for each type of particles, e.g., [0.2, 0.2] GeV
        particle_type: 
            should be consistent with initial_distribution, masses and charges, indicate whether the particle is 
            classical (0), fermi (1) or bosonic type (2), e.g., for five particle species, [0,0,1,1,2]
            means these five particles are [classical, classical, fermi, fermi, bosonic]
        charges:
            charges for each type of particles, e.g., [0.2, 0.2]
        '''
        
        super().__init__(dx, dy, dz, x_left_bound, y_left_bound, z_left_bound, \
                         dpx, dpy, dpz, px_left_bound, py_left_bound, pz_left_bound, \
                         x_grid_size, y_grid_size, z_grid_size, \
                         px_grid_size, py_grid_size, pz_grid_size, \
                         initial_distribution, masses, particle_type, charges)
        
    def obtain_collision_term(self, flavor, collision_type, dF, CA, CF, dA, Ng, Nq, g, i_time_step, num_samples = 100):
        '''
        Method for obtaining the collision term. This is to be modified by users.

        Params
        ======
        flavor: 
            all possible collisions for the given final particle, type: numpy array, eg: for final d, we have
            ud->ud (0), dd->dd (1), sd->sd (0), u_bar+d->u_bar+d (0), s_bar+d->s_bar+d (0), gd->gd (5)
            d_bar+d->d_bar+d (2), u_bar+u->d_bar+d (3), s_bar+s->d_bar+d (3), gg->d_bar+d (4)
            flavor=np.array([[[0,1,0,1],[1,1,1,1],[2,1,2,1],[3,1,3,1],[5,1,5,1],[6,1,6,1],\
                              [4,1,4,1],[3,0,4,1],[5,2,4,1],[6,6,4,1]]],dtype=np.int64)
        collision_type: 
            an index for flavor, type: numpy array, eg: collision_type=np.array([[0,1,0,0,0,5,2,3,3,4]],dtype=np.int64)
        dF, CA, CF, dA:
            color factors in SU(3)
        Ng, Nq:
            degenecy factors for quarks and gluons
        g:
            coupling constant
        i_time_step:
            the current time step, start from 1 to n_step-1
        num_samples = 100:
            number of samples in Monte Carlo integrations
        '''
        
        # initialize force term
        collision_term = cupy.zeros_like(self.f, dtype = np.float64)
        
        self.i_time_step = i_time_step
        # only initialize for the first time step
        if i_time_step == 1:

            self.flavor, self.collision_type = cupy.asarray(flavor), cupy.asarray(collision_type)

            # seeds for random generator
            self.rng_states = cuda.to_device(create_xoroshiro128p_states(self.threadsperblock * self.blockspergrid_total,
                                        seed=random.sample(range(0,1000),1)[0]))

        # collisional term must be multipiled by the integration volume and divided by the number of samples
        # half_px*half_py*half_pz*half_py*half_pz*2**5*1/(2*PI)**5/2**4/num_samples
        coef = abs((self.px_left_bound)*(self.py_left_bound)*(self.pz_left_bound)*\
                   (self.py_left_bound)*(self.pz_left_bound)*2/((2*math.pi)**5)/(num_samples*4))

        # call force kernel for the force term
        collision_term_kernel[self.blockspergrid_total, self.threadsperblock](self.f, \
                                                                              self.masses, \
                                                                              self.total_grid, \
                                                                              self.num_of_particle_types, \
                                                                              self.px_grid_size, self.py_grid_size, self.pz_grid_size, \
                                                                              self.x_grid_size, self.y_grid_size, self.z_grid_size, \
                                                                              self.px_left_bound, self.py_left_bound, self.pz_left_bound, \
                                                                              self.dpx, self.dpy, self.dpz, \
                                                                              self.flavor, self.collision_type, \
                                                                              num_samples, self.rng_states, \
                                                                              g, dF, CA, CF, dA, Nq, Ng, \
                                                                              collision_term, coef)

        self.collision_term = collision_term
        
        #return cupy.asnumpy(self.collision_term)
    
    def obtain_current_particle_numbers(self, Nq, Ng):
        '''
        Return the particle numbers of the current distributions.
        
        Return
        ======
        [num_u_quark, num_d_quark, num_s_quark, num_u_bar_quark, num_d_bar_quark, num_s_bar_quark, gluon]'''
        
        # the volume element associated the integration
        phase_space_volume_element = self.dx*self.dy*self.dz*self.dpz*self.dpy*self.dpz/(2*math.pi)**3
        
        return cupy.asnumpy(cupy.sum(self.f, axis = [1,2,3,4,5,6]))*phase_space_volume_element*np.array([Nq,Nq,Nq,Nq,Nq,Nq,Ng])
    
    def obtain_rho_J(self, Nq, Ng):
        '''
        Return the charge density and current density of the current distribution. This is to provide rho and J to EMsolver.
        
        Params
        ======
        Nq, Ng: degeneracy factors for quarks and gluons.
        '''
        
        # allocate a space in GPU to save rho and J
        rho = cupy.zeros([self.x_grid_size, self.y_grid_size, self.z_grid_size], dtype = np.float64)
        J = cupy.zeros([self.x_grid_size, self.y_grid_size, self.z_grid_size, 3], dtype = np.float64)
        
        # the volume element associated the integration
        momentum_space_volume_element = self.dpx*self.dpy*self.dpz/(2*math.pi)**3
        
        # call force kernel for the force term
        rho_J_kernel[self.blockspergrid_total, self.threadsperblock](self.f, rho, J, self.masses, self.charges,\
                                                                     self.total_grid, self.num_of_particle_types, \
                                                                     self.px_grid_size, self.py_grid_size, self.pz_grid_size, \
                                                                     self.x_grid_size, self.y_grid_size, self.z_grid_size, \
                                                                     self.px_left_bound, self.py_left_bound, self.pz_left_bound, \
                                                                     self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz, \
                                                                     Nq, Ng)
        
        
        
        self.rho, self.J = rho*momentum_space_volume_element, J*momentum_space_volume_element
        
    def acquire_rho_J(self):
        '''
        Acquire rho and J to CPU memory.
        
        Return
        ======
        (rho, J):
            rho is of size [nx, ny, nz], J is of size [nx, ny, nz, 3]'''
        
        return cupy.asnumpy(self.rho), cupy.asnumpy(self. J)
    
    def add_time_step(self, time_step):
        self.i_time_step = time_step
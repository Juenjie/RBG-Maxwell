import cupy, ray, math, random, os
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from .check_legacy_of_distribution import check_legacy_of_distributions
from .cuda_kernels.EM_force_kernel import EM_force_kernel
from .cuda_kernels.force_term_kernel import force_term_kernel
from .cuda_kernels.velocity_term_kernel import velocity_term_kernel

class BEsolver():
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

        self.particle_type = particle_type
        
        # save the variables in the class
        self.dx, self.dy, self.dz = dx, dy, dz
        self.x_left_bound, self.y_left_bound, self.z_left_bound = x_left_bound, y_left_bound, z_left_bound
        self.x_grid_size, self.y_grid_size, self.z_grid_size = x_grid_size, y_grid_size, z_grid_size
        self.dpx, self.dpy, self.dpz = dpx, dpy, dpz
        self.px_left_bound, self.py_left_bound, self.pz_left_bound = px_left_bound, py_left_bound, pz_left_bound
        self.px_grid_size, self.py_grid_size, self.pz_grid_size = px_grid_size, py_grid_size, pz_grid_size   
        
        # total grid numbers
        self.total_grid = x_grid_size*y_grid_size*z_grid_size*px_grid_size*py_grid_size*pz_grid_size   
        self.spatial_total_grid = x_grid_size*y_grid_size*z_grid_size
        
        # Configure the blocks
        self.threadsperblock = 32
        # configure the grids
        self.blockspergrid_total = (self.total_grid + (self.threadsperblock - 1)) // self.threadsperblock
        self.blockspergrid_spatial_total = (self.spatial_total_grid + (self.threadsperblock - 1)) // self.threadsperblock

        # copy initial distribution functions to GPU, size [n_type, nx, ny, nz, npx, npy, npz]
        self.f = cupy.asarray(initial_distribution)
        
        # give zero values to these terms such that when not calling upon, zero value will be used without occupying the memory
        self.collision_term = 0.
        self.velocity_term = 0.
        self.force_term = 0.
        
        # masses and charges of the particles
        self.masses = cupy.asarray(masses)
        self.charges = cupy.asarray(charges)
        
        # total number of particle types
        self.num_of_particle_types = len(masses)
        
        print("Using GPU {} for BE calculation.".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        
    def update_f(self, dt, rescale_of_dt = 0.1, maximum_count = 3, count = 0):
        '''
        This method should be called after the evaluation of the 
        collisional term, velocity term and force term
        
        Params
        ======
        dt: 
            time difference
        rescale_of_dt = 0.5: 
            new dt should be smaller, dt = dt * rescale_of_dt
        maximum_count = 3: 
            indicate how many rescales should be used. After each rescale of dt, it will be 
            shrinked by a factor of rescale_of_dt.
            
        Return
        ======
        dt:
            the current value of dt used in this time step
        '''
        
        # update the distribution according to dt        
        self.f_temp = cupy.zeros_like(self.f, dtype = np.float64)
        # for velocity dt has been multiplied in the calculation
        self.f_temp = self.f + (self.collision_term - self.velocity_term - self.force_term)*dt

        # if the newly calculated distributions are incorrect, then decrease dt and recalculate till legal
        trueOrfalse,b,c,d = check_legacy_of_distributions(self.f_temp, self.particle_type)
        if trueOrfalse == False:
            print('rescale for ',count,' depth at time_step: ',self.i_time_step,' where ',b,' equals ',c,' for particle ',d)
            
            # return error if count is larger than maximum_count
            if count > maximum_count:
                raise AssertionError("The algorithm fails to give correct distribution with maximum_count = {}. Try increase maximum_count or choose a proper initial distribution.".format(maximum_count))
                
            # rescale dt and recalculate
            dt = dt * rescale_of_dt
            count += 1
            self.update_f(dt = dt, rescale_of_dt = rescale_of_dt, count = count)
            
            

        return dt, count
    
    def obtain_velocity_term(self, boundary_info_x, boundary_info_y, boundary_info_z, dt):
        '''
        Method for obtaining the velocity term.
        boundary_info_x: of shape [n_type, ny, nz, 2]
        boundary_info_y: of shape [n_type, nz, nx, 2]
        boundary_info_z: of shape [n_type, nx, ny, 2]'''

        # initialize velocity term
        velocity_term = cupy.zeros_like(self.f, dtype = np.float64)
        
        boundary_info_x, boundary_info_y, boundary_info_z = \
                                                            cupy.asarray(boundary_info_x),\
                                                            cupy.asarray(boundary_info_y),\
                                                            cupy.asarray(boundary_info_z)
        
        # call velocity kernel for the velocity term
        velocity_term_kernel[self.blockspergrid_total, self.threadsperblock](self.f, \
                                                                             velocity_term, self.masses, \
                                                                             self.total_grid, self.num_of_particle_types, \
                                                                             self.px_grid_size, self.py_grid_size, self.pz_grid_size, \
                                                                             self.x_grid_size, self.y_grid_size, self.z_grid_size, \
                                                                             self.px_left_bound, self.py_left_bound, self.pz_left_bound, \
                                                                             dt, self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz,\
                                                                             boundary_info_x, boundary_info_y, boundary_info_z, self.i_time_step-1)
        self.velocity_term = velocity_term

        #return cupy.asnumpy(self.velocity_term)
        
    def obtain_force_term(self):
        '''
        The method for obtaining the froce term. Provide external_force before calling this method.'''
        
        # initialize force term and external_froce
        force_term = cupy.zeros_like(self.f, dtype = np.float64)
        
        # call force kernel for the force term
        force_term_kernel[self.blockspergrid_total, self.threadsperblock](self.f, \
                                                                          force_term, self.external_force, \
                                                                          self.total_grid, self.num_of_particle_types, \
                                                                          self.px_grid_size, self.py_grid_size, self.pz_grid_size, \
                                                                          self.x_grid_size, self.y_grid_size, self.z_grid_size, \
                                                                          self.dpx, self.dpy, self.dpz)
        self.force_term = force_term
        
        #return cupy.asnumpy(self.force_term)
        
    def copy_f_temp_to_f(self):
        '''
        If the newly calculated temp_f is legal then copy it to f. Only calling this method after the confirmation of dt. 
        The method should be called after update_f() method.'''
        
        self.f = self.f_temp
        
    def reset_f(self, distribution):
        '''
        Use a new distribution to replace the current one.'''
        
        self.f = cupy.asarray(distribution)
        
    def acquire_f(self):
        '''
        Return self.f as numpy arrays. Acquire the current distribution function, otherwise the values will be overwritten.'''
        
        return cupy.asnumpy(self.f)
    
    def obtain_EM_force(self, E_field, B_field):
        '''
        Given distributions of E and B, and find the EM forces at each phase point. Call this method before obtain_force_term.
        
        Params
        ======
        E_field, B_field:
            electric and magnetic field at each spatial point, of size [3, nx, ny, nz, ]
        '''
        
        # copy E and B to GPU
        E_field, B_field = cupy.asarray(E_field), cupy.asarray(B_field)
        
        # allocate a space in GPU to save the forces
        external_force = cupy.zeros([self.num_of_particle_types, self.x_grid_size, self.y_grid_size, self.z_grid_size, \
                                     self.px_grid_size, self.py_grid_size, self.pz_grid_size, 3], dtype = np.float64)
        
        # call force kernel for the force term
        EM_force_kernel[self.blockspergrid_total, self.threadsperblock](external_force, self.masses, self.charges,\
                                                                        self.total_grid, self.num_of_particle_types, \
                                                                        self.px_grid_size, self.py_grid_size, self.pz_grid_size, \
                                                                        self.x_grid_size, self.y_grid_size, self.z_grid_size, \
                                                                        self.px_left_bound, self.py_left_bound, self.pz_left_bound, \
                                                                        self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz, \
                                                                        E_field, B_field)
        
        
        self.external_force = external_force
        
        #return cupy.asnumpy(self.external_force)

from numba import cuda
import numpy as np
import math
import cupy
from .cuda_kernel import drift_force_term_kernel

def Drift_Vlasov_terms(f_x_p_t, Fx, Fy, Fz, \
                       masses, total_grid, num_of_particle_types, \
                       npx, npy, npz, nx, ny, nz, \
                       half_px, half_py, half_pz, \
                       dx, dy, dz, dpx, dpy, dpz, number_momentum_levels,\
                       x_bound_config, y_bound_config, z_bound_config, \
                       blockspergrid_total_phase, threadsperblock,\
                       collision_term, dt, c, current_time_step, \
                       whether_drift, whether_Vlasov, drift_order):
    """Give the Vlasov terms of the given distribution.
    
    Params
    ======
    f_x_p_t:
        distribution function for all particles at all levels. The shape of f corresponds to
        [momentum_levels, particle_species, nx*ny*nz*npx*npy*npz].
    velocity_term:
        velocity term at each phase space point, of the same shape as f_x_p_t
    force_term:
        force term at each phase space point, of the same shape as f_x_p_t
    Fx, Fy, Fz: 
        external forces are of shape
        Fx/Fy/Fz: [particle_species, nx*ny*nz*npx*npy*npz]
    masses: 
        masses of the particles
    total_grid:
        total number of grids, nx*ny*nz*npx*npy*npz
    num_of_particle_types:
        total number of particle types, this is len(masses)
    npx, npy, npz: 
        number of grid sizes in in momentum domain, e.g., [5,5,5]
    nx, ny, nz:
        number of grids in spatial domain, e.g., [5, 5, 5]
    half_px, half_py, half_pz:
        the value of half momentum domain, in unit Energy   
        this is an array containing the values for different type of particles 
    dx, dy, dz: 
        infinitesimal difference in spatial coordinate, these are in Energy^-1 (recomended)        
    dpx, dpy, dpz:
        infinitesimal difference in momentum coordinate, these are in Energy (recomended)
        this is an array containing the values for different type of particles
    number_momentum_levels: 
        how many momentum levels are used for the particles
    x_bound_config, y_bound_config, z_bound_config:
        configuretions of the boundary conditions. 
        x_bound_config is of shape [ny, nz]
        y_bound_config is of shape [nz, nx]
        z_bound_config is of shape [nx, ny]
        values of each component (between 0~1) corresponds to the component being reflected
        1. indicates absorption and 0. indicates relection.
    blockspergrid_total_phase, threadsperblock:
        block and thread sizes for GPU calculation
    whether_drift, whether_Vlasov:
        indication of whether to calculate Vlasov or Drift term
        """
    
    # at each time step try second firstly
    f_updated = cupy.zeros_like(f_x_p_t)
    f_updated[:, 1, :] = f_x_p_t[:, 1, :]

    drift_force_term_kernel[blockspergrid_total_phase, threadsperblock]\
    (f_x_p_t, Fx, Fy, Fz, \
     masses, total_grid, num_of_particle_types, \
     npx, npy, npz, nx, ny, nz,\
     half_px, half_py, half_pz, \
     dx, dy, dz, dpx, dpy, dpz, number_momentum_levels,\
     x_bound_config, y_bound_config, z_bound_config, \
     f_updated, collision_term, \
     dt, c, current_time_step, whether_drift, whether_Vlasov,\
     drift_order)
    
    return f_updated

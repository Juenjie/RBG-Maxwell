from numba import cuda
import math
import cupy
from .cuda_kernel import EM_force_kernel

"""Give the external forces in the Vlasov terms"""
    
def External_forces(masses, charges,\
                    total_phase_grids, num_particle_species, \
                    npx,npy,npz,nx,ny,nz, \
                    half_px, half_py, half_pz, \
                    dx, dy, dz, dpx, dpy, dpz, \
                    blockspergrid_total_phase, threadsperblock, number_momentum_levels,\
                    Ex, Ey, Ez, Bx, By, Bz, c):
    """Give the electromagnetic forces at each phase point.
     Note, here the EMsolver generates E and B for all GPUs, and only the correct
     E and B should be used in this class.
    Params
    ======
    Fx, Fy, Fz: 
        external forces are of shape
        Fx/Fy/Fz: [momentum_level, particle_species, nx*ny*nz*npx*npy*npz]
    masses: 
        the mass of the particles, cp-array, the order should follow the particles in distribution function
    charges: 
        the charge of the particles, cp-array, the order should follow the particles in distribution function
    total_phase_grids:
        nx*ny*nz*npx*npy*npz
    num_particle_species:
        total number of particle species
    npx, npy, npz: 
        number of momentum grids
    nx, ny, nz: 
        number of spatial grids 
    half_px, half_py, half_pz:
    the value of half momentum domain, in unit Energy   
    this is an array containing the values for different type of particles 
    dx, dy, dz: 
        infinitesimal difference in spatial coordinate, these are in Energy^-1 (recomended)        
    dpx, dpy, dpz:
        infinitesimal difference in momentum coordinate, these are in Energy (recomended)
        this is an array containing the values for different type of particles     
    blockspergrid_total_phase, threadsperblock:
        block and thread sizes for GPU calculation
    number_momentum_levels:
        how many momentum levels are used for the particles
    Ex, Ey, Ez, Bx, By, Bz: 
        eletromagnetic fields for the current GPU."""

    
    # allocate a space in GPU to save the electromagnetic forces
    Fx = cupy.empty([number_momentum_levels, num_particle_species, total_phase_grids])
    Fy = cupy.empty([number_momentum_levels, num_particle_species, total_phase_grids])
    Fz = cupy.empty([number_momentum_levels, num_particle_species, total_phase_grids])

    EM_force_kernel[blockspergrid_total_phase, threadsperblock]\
    (Fx, Fy, Fz, masses, charges,\
     total_phase_grids, num_particle_species, \
     npx,npy,npz,nx,ny,nz, \
     half_px, half_py, half_pz, \
     dx, dy, dz, dpx, dpy, dpz, \
     Ex, Ey, Ez, Bx, By, Bz, number_momentum_levels, c)
    
    return Fx, Fy, Fz
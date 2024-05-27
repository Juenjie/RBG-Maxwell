from numba import cuda
import numpy as np
import math
import cupy
from .cuda_kernel import rho_J_kernel, electric_rho_J_kernel

"""Give the macroscopic quantities such as number density, charge density.
The distribution function f is of shape 
[momentum_levels, particle_species, nx*ny*nz*npx*npy*npz]"""

def density(f, num_particle_species, total_spatial_grids, \
            masses, charges, \
            total_phase_grids, momentum_volume_element, npx, npy, npz, \
            nx, ny, nz, half_px, half_py, half_pz, \
            dx, dy, dz, dpx, dpy, dpz, num_momentum_levels,\
            blockspergrid_total_phase, threadsperblock, c):
    """Give the current density of the particles in spatial domain.
    params
    ======
    f:
        distribution function f is of shape 
        [momentum_levels, particle_species, nx*ny*nz*npx*npy*npz]
    num_particle_species:
        total number of particle species
    total_spatial_grids:
        total spatial grids = nx*ny*nz
    rho, Jx, Jy, Jz: 
        particle density and current densities
    masses: 
        the mass of the particles, cp-array, the order should follow the particles in distribution function
    charges: 
        the charge of the particles, cp-array, the order should follow the particles in distribution function
    total_phase_grids:
        nx*ny*nz*npx*npy*npz
    momentum_volume_element:
        dpx*dpy*dpz
    npx, npy, npz: 
        number of momentum grids
    nx, ny, nz: 
        number of spatial grids 
    half_px, half_py, half_pz:
        the value of half of momentum domain
        this is an array containing the values for different type of particles 
    dx, dy, dz: 
        infinitesimal difference in spatial domain
        this is an array containing the values for different type of particles 
    dpx, dpy, dpz: 
        infinitesimal difference in momentum domain    
    blockspergrid_total_phase, threadsperblock:
        block and thread sizes for GPU calculation
    num_momentum_levels:
        number of momentum levels
    c:
        numerical value of the speed of light in FU (Flexible Unit)
    """

    
    rho = cupy.zeros([num_particle_species, total_spatial_grids]) 
    Jx = cupy.zeros([num_particle_species, total_spatial_grids]) 
    Jy = cupy.zeros([num_particle_species, total_spatial_grids]) 
    Jz = cupy.zeros([num_particle_species, total_spatial_grids])

    rho_J_kernel[blockspergrid_total_phase, threadsperblock]\
    (f, rho, Jx, Jy, Jz, masses, charges,\
     total_phase_grids, num_particle_species, total_spatial_grids,\
     momentum_volume_element, npx, npy, npz, \
     nx, ny, nz, num_momentum_levels,\
     half_px, half_py, half_pz, \
     dx, dy, dz, dpx, dpy, dpz, c)   

    return rho, Jx, Jy, Jz

def charged_density(f, num_particle_species, total_spatial_grids, \
                    masses, charges, \
                    total_phase_grids, momentum_volume_element, npx, npy, npz, \
                    nx, ny, nz, half_px, half_py, half_pz, \
                    dx, dy, dz, dpx, dpy, dpz, num_momentum_levels,\
                    blockspergrid_total_phase, threadsperblock, c):
    """Give the electric current of the particles in spatial domain
    params
    ======
    f:
        distribution function f is of shape 
        [momentum_levels, particle_species, nx*ny*nz*npx*npy*npz]
    c:
        numerical value of the speed of light in FU (Flexible Unit)
    num_particle_species:
        total number of particle species
    total_spatial_grids:
        total spatial grids = nx*ny*nz
    electric_rho, electric_Jx, electric_Jy, electric_Jz: 
        electric density and electric current densities
    masses: 
        the mass of the particles, cp-array, the order should follow the particles in distribution function
    charges: 
        the charge of the particles, cp-array, the order should follow the particles in distribution function
    total_phase_grids:
        nx*ny*nz*npx*npy*npz
    momentum_volume_element:
        dpx*dpy*dpz
    npx, npy, npz: 
        number of momentum grids
    nx, ny, nz: 
        number of spatial grids 
    half_px, half_py, half_pz:
        the value of half momentum domain
        this is an array containing the values for different type of particles 
    dx, dy, dz: 
        infinitesimal difference in spatial domain
    dpx, dpy, dpz: 
        infinitesimal difference in momentum domain 
        this is an array containing the values for different type of particles 
    blockspergrid_total_phase, threadsperblock:
        block and thread sizes for GPU calculation
    num_momentum_levels:
        number of momentum levels
    c:
        numerical value of the speed of light in FU (Flexible Unit)
    """ 
    
    electric_rho = cupy.zeros([total_spatial_grids])  
    electric_Jx = cupy.zeros([total_spatial_grids]) 
    electric_Jy = cupy.zeros([total_spatial_grids]) 
    electric_Jz = cupy.zeros([total_spatial_grids]) 

    # print('before fFe:', f.reshape([num_momentum_levels, num_particle_species, nx, ny, nz, npx, npy, npz])[0,0,:,50,0].max(), f.reshape([num_momentum_levels, num_particle_species, nx, ny, nz, npx, npy, npz])[0,0,50,:,0].max())
    electric_rho_J_kernel[blockspergrid_total_phase, threadsperblock]\
    (f, electric_rho, electric_Jx, electric_Jy, electric_Jz, \
     masses, charges,\
     total_phase_grids, num_particle_species, total_spatial_grids,\
     momentum_volume_element, npx, npy, npz, nx, ny, nz, \
     half_px, half_py, half_pz, \
     dx, dy, dz, dpx, dpy, dpz, num_momentum_levels, c)    
    
#     print('rho max: ', electric_rho.min())
#     print('f max:', f.max())
#     print(dx, dy, dz, dpx, dpy, dpz)
#     print(momentum_volume_element*charges)
    # print('after JFe:', electric_rho.max(), electric_Jx.max(), electric_Jy.max(), electric_Jz.max())
    return electric_rho, electric_Jx, electric_Jy, electric_Jz
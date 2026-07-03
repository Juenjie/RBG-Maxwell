from numba import cuda
import math

'''
CUDA kernel for obtaining rho and J from given distributions, masses and charges.
This function is purely an integration involving the distributions.

Params
======
f_x_p_t:
    distribution of the particles, of shape [n_type, nx, ny, nz, npx, npy, npz]
rho, J:
    charge and current density at each spatial grid, rho is of size [nx, ny, nz], J is of size [nx, ny, nz, 3]
masses:
    masses for each type of particles, e.g., [0.2, 0.2] GeV
charges:
    charges for each type of particles, e.g., [0.2, 0.2]
total_grid:
    total number of grids, nx*ny*nz*npx*npy*npz
num_of_particle_types:
    total number of particle types, this is len(masses)
px_grid_size, py_grid_size, pz_grid_size: 
    number of grid sizes in in momentum domain, e.g., [5,5,5]
x_grid_size, y_grid_size, z_grid_size:
    number of grids in spatial domain, e.g., [5, 5, 5]
px_left_bound, py_left_bound, pz_left_bound:
    left boundary of the momentum region, in unit GeV          
dx, dy, dz: 
    infinitesimal difference in spatial coordinate, these are in GeV^-1 (recomended)        
dpx, dpy, dpz:
    infinitesimal difference in momentum coordinate, these are in GeV (recomended)
momentum_space_volume_element:
    dpz*dpy*dpz/(2*math.pi)**3
'''

@cuda.jit
def rho_J_kernel(f_x_p_t, rho, J, masses, charges,\
                 total_grid, num_of_particle_types, \
                 px_grid_size, py_grid_size, pz_grid_size, \
                 x_grid_size, y_grid_size, z_grid_size, \
                 px_left_bound, py_left_bound, pz_left_bound, \
                 dx, dy, dz, dpx, dpy, dpz, Nq, Ng):
    
    # threads loop in one dimension
    i_grid = cuda.grid(1)
            
    if i_grid < total_grid:
        
        # convert one-d index into six-d
        ix = i_grid%x_grid_size
        ix_rest = i_grid//x_grid_size
        iy = ix_rest%y_grid_size
        iy_rest = ix_rest//y_grid_size
        iz = iy_rest%z_grid_size
        iz_rest = iy_rest//z_grid_size
        ipx = iz_rest%px_grid_size
        ipx_rest = iz_rest//px_grid_size
        ipy = ipx_rest%py_grid_size
        ipy_rest = ipx_rest//py_grid_size
        ipz = ipy_rest%pz_grid_size
        
        # acquire p from the central value
        px = (ipx+0.5)*dpx + px_left_bound
        py = (ipy+0.5)*dpy + py_left_bound
        pz = (ipz+0.5)*dpz + pz_left_bound
        
        # loop through all species
        for p_type in range(num_of_particle_types):
            
            # energy for current grid
            mp_squared = masses[p_type]**2
            Ep = math.sqrt(mp_squared+px**2+py**2+pz**2)
            
            # charge and gamma for the current particle
            charge  = charges[p_type]
            gamma = Ep/masses[p_type]
            
            # distribution function multiplied by change and gamma
            if p_type == 6:
                modified_f = f_x_p_t[p_type, ix, iy, iz, ipx, ipy, ipz]*gamma*charge*Ng
            else:
                modified_f = f_x_p_t[p_type, ix, iy, iz, ipx, ipy, ipz]*gamma*charge*Nq
            
            # vx, vy, vz
            vx = px/Ep
            vy = py/Ep
            vz = pz/Ep
            
            # rho and J at each phase space point
            cuda.atomic.add(rho, (ix,iy,iz), modified_f)
            cuda.atomic.add(J, (ix,iy,iz,0), modified_f*vx)
            cuda.atomic.add(J, (ix,iy,iz,1), modified_f*vy)
            cuda.atomic.add(J, (ix,iy,iz,2), modified_f*vz)
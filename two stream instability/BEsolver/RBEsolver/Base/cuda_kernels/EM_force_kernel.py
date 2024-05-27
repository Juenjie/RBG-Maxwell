from numba import cuda
import math

'''
CUDA kernel for obtaining electric and magnetic force at each phase space for a spcific particle type.
F = q*(E + V \cross B)

Params
======
external_force:
    external force receieved by the particles, of shape [num_particle_types, nx, ny, nz, npx, npy, npz, 3], GeV^2
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
E_field, B_feild:
    electric and magnetic field at each spatial point, of size [3, nx, ny, nz, ]
'''

@cuda.jit
def EM_force_kernel(external_force, masses, charges,\
                    total_grid, num_of_particle_types, \
                    px_grid_size, py_grid_size, pz_grid_size, \
                    x_grid_size, y_grid_size, z_grid_size, \
                    px_left_bound, py_left_bound, pz_left_bound, \
                    dx, dy, dz, dpx, dpy, dpz, \
                    E_field, B_field):
    
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
        
        # E and B, this should be constant with the array [Ex, Ey, Ez] in the bench mark
        Efx = E_field[0, ix, iy, iz]
        Efy = E_field[1, ix, iy, iz]
        Efz = E_field[2, ix, iy, iz]
        Bfx = B_field[0, ix, iy, iz]
        Bfy = B_field[1, ix, iy, iz]
        Bfz = B_field[2, ix, iy, iz]
        
        # loop through all species
        for p_type in range(num_of_particle_types):
            
            # energy for current grid
            mp_squared = masses[p_type]**2
            Ep = math.sqrt(mp_squared+px**2+py**2+pz**2)
            
            # charge for the current particle
            charge  = charges[p_type]
            
            # vx, vy, vz
            vx = px/Ep
            vy = py/Ep
            vz = pz/Ep

            # external EM force
            external_force[p_type,ix,iy,iz,ipx,ipy,ipz,0] = charge*(Efx + (vy*Bfz - vz*Bfy))
            external_force[p_type,ix,iy,iz,ipx,ipy,ipz,1] = charge*(Efy + (vz*Bfx - vx*Bfz))
            external_force[p_type,ix,iy,iz,ipx,ipy,ipz,2] = charge*(Efz + (vx*Bfy - vy*Bfx))
                
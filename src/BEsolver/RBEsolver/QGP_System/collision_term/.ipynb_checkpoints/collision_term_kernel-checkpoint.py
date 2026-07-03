from numba import float64
import math
from numba import cuda 
from .collision_term_device import *

'''
CUDA kernel for obtaining the collision term using Monte Carlo integration.

f_x_p_t: 
    distributions of size [n_type, nx, ny, nz, npx, npy, npz]
collision_term: 
    this is the term that needs to be evaluated, of the same size as f_x_p_t
masses:
    masses for each type of particles, e.g., [0.2, 0.2] GeV
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
flavor: 
    all possible collisions for the given final particle, type: numpy array, eg: 
    for final d, we have
    ud->ud (0), dd->dd (1), sd->sd (0), u_bar+d->u_bar+d (0), s_bar+d->s_bar+d (0), gd->gd (5)
    d_bar+d->d_bar+d (2), u_bar+u->d_bar+d (3), s_bar+s->d_bar+d (3), gg->d_bar+d (4)
    flavor=np.array([[[0,1,0,1],[1,1,1,1],[2,1,2,1],[3,1,3,1],[5,1,5,1],[6,1,6,1],\
                      [4,1,4,1],[3,0,4,1],[5,2,4,1],[6,6,4,1]]],dtype=np.int64)
collision_type: 
    an index for flavor, type: numpy array, eg:
    collision_type=np.array([[0,1,0,0,0,5,2,3,3,4]],dtype=np.int64)
dF, CA, CF, dA, Ng, Nq, g:
    float64, physical constants
num_samples:
    int, number of samples for the five dimensional integration for collision term.
rng_states:
    array for generate random samples.
coef:
    half_px*half_py*half_pz*half_py*half_pz*2**5*1/(2*PI)**5/2**4/num_samples
'''

@cuda.jit
def collision_term_kernel(f_x_p_t, masses, \
                          total_grid, num_of_particle_types, \
                          px_grid_size, py_grid_size, pz_grid_size, \
                          x_grid_size, y_grid_size, z_grid_size, \
                          px_left_bound, py_left_bound, pz_left_bound, \
                          dpx, dpy, dpz, \
                          flavor, collision_type, \
                          num_samples, rng_states, \
                          g, dF, CA, CF, dA, Nq, Ng, \
                          collision_term, coef):
    
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
        
        # mass_squared for all particle types
        # must be constant with u,d,s,ubar,dbar,sbar,gluon
        masses_squared_collect = (masses[0]**2, masses[1]**2, masses[2]**2, masses[3]**2, masses[4]**2, masses[5]**2, masses[6]**2)
        mg_regulator_squared = masses_squared_collect[6]
        
        # loop through all species
        for p_type in range(num_of_particle_types):
            
            # energy for current grid
            mp_squared = masses_squared_collect[p_type]
            Ep = math.sqrt(mp_squared+px**2+py**2+pz**2)
            
            # collision term
            # evaluate collision term  at certain phase point
            collision_term_at_specific_point(f_x_p_t, flavor[p_type], collision_type[p_type],
                                             masses_squared_collect, num_samples,
                                             -px_left_bound, -py_left_bound, -pz_left_bound,
                                             rng_states, i_grid, mp_squared,
                                             ix, iy, iz, ipx, ipy, ipz, dpx, dpy, dpz,
                                             g, dF, CA, CF, dA, Nq, Ng, mg_regulator_squared, p_type,
                                             collision_term, px, py, pz, Ep, coef)
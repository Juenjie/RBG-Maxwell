from numba import cuda
import math

'''
CUDA kernel for obtaining the force term using upwind difference.
Peoriodical boundary conditions are used for momentum distributions.

f_x_p_t:
    distribution of the particles, of shape [n_type, nx, ny, nz, npx, npy, npz]
force_term:
    force term at each phase space point, of the same shape as f_x_p_t
external_force:
    external force receieved by the particles, of shape [num_particle_types, nx, ny, nz, npx, npy, npz, 3], GeV^2
total_grid:
    total number of grids, nx*ny*nz*npx*npy*npz
num_of_particle_types:
    total number of particle types, this is len(masses)
px_grid_size, py_grid_size, pz_grid_size: 
    number of grid sizes in in momentum domain, e.g., [5,5,5]
x_grid_size, y_grid_size, z_grid_size:
    number of grids in spatial domain, e.g., [5, 5, 5]
dpx, dpy, dpz:
    infinitesimal difference in momentum coordinate, these are in GeV (recomended)
'''

@cuda.jit(device=True)
def pf_pp(Fx, dx, fleftx, fcurrent, frightx):
    if Fx > 0:
        return (fcurrent - fleftx)/dx
    return (frightx - fcurrent)/dx

@cuda.jit
def force_term_kernel(f_x_p_t, force_term, external_force, \
                      total_grid, num_of_particle_types, \
                      px_grid_size, py_grid_size, pz_grid_size, \
                      x_grid_size, y_grid_size, z_grid_size, \
                      dpx, dpy, dpz):
    
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
        
        # loop through all species
        for p_type in range(num_of_particle_types):

            # enforce periodical boundary conditions, ipxPlus --> ipx+1
            ipxPlus = (ipx+1)%px_grid_size
            ipyPlus = (ipy+1)%py_grid_size
            ipzPlus = (ipz+1)%pz_grid_size

            # -1%3 should be 2, but cuda yields 0, so we use
            ipxMinus = (ipx+px_grid_size-1)%px_grid_size
            ipyMinus = (ipy+py_grid_size-1)%py_grid_size
            ipzMinus = (ipz+pz_grid_size-1)%pz_grid_size

            # distribution functions at p, p-dpx, px+dpx
            fcurrentp = f_x_p_t[p_type, ix, iy, iz, ipx, ipy, ipz]
            fleftpx = f_x_p_t[p_type, ix, iy, iz, ipxMinus, ipy, ipz]
            frightpx = f_x_p_t[p_type, ix, iy, iz, ipxPlus, ipy, ipz]
            fleftpy = f_x_p_t[p_type, ix, iy, iz, ipx, ipyMinus, ipz]
            frightpy = f_x_p_t[p_type, ix, iy, iz, ipx, ipyPlus, ipz]
            fleftpz = f_x_p_t[p_type, ix, iy, iz, ipx, ipy, ipzMinus]
            frightpz = f_x_p_t[p_type, ix, iy, iz, ipx, ipy, ipzPlus]
            
            # External forces at p, p-dpx, px+dpx
            Fcurrentpx = external_force[p_type, ix, iy, iz, ipx, ipy, ipz, 0]
            Fcurrentpy = external_force[p_type, ix, iy, iz, ipx, ipy, ipz, 1]
            Fcurrentpz = external_force[p_type, ix, iy, iz, ipx, ipy, ipz, 2]
            
            # force term
            # F.grad(f,p) = div.(Ff)
            # upwind difference
            #force_term[p_type,ix,iy,iz,ipx,ipy,ipz] = pf_pp(Fcurrentpx, dpx, Fleftpx*fleftpx, Fcurrentpx*fcurrentp, Frightpx*frightpx) + pf_pp(Fcurrentpy, dpy, Fleftpy*fleftpy, Fcurrentpy*fcurrentp, Frightpy*frightpy) + pf_pp(Fcurrentpz, dpz, Fleftpz*fleftpz, Fcurrentpz*fcurrentp, Frightpz*frightpz)
 
            # this is amazingly stable
            force_term[p_type,ix,iy,iz,ipx,ipy,ipz] = Fcurrentpx*pf_pp(Fcurrentpx, dpx, fleftpx, fcurrentp, frightpx) + Fcurrentpy*pf_pp(Fcurrentpy, dpy, fleftpy, fcurrentp, frightpy) + Fcurrentpz*pf_pp(Fcurrentpz, dpz, fleftpz, fcurrentp, frightpz)
                
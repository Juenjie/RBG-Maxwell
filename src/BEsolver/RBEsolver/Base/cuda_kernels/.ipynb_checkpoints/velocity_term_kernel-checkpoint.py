from numba import cuda
import math

'''
CUDA kernel for obtaining the velocity term using upwind difference.
The device function is to take the upwind difference according to the values of v.
Peoriodical boundary conditions are used for spatial distributions. 

f_x_p_t:
    distribution of the particles, of shape [n_type, nx, ny, nz, npx, npy, npz]
velocity_term:
    velocity term at each phase space point, of the same shape as f_x_p_t masses, \
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
'''

@cuda.jit(device=True)
def mirror(ngrid, i):
    middle_grid = int(ngrid/2)
    return int(middle_grid + middle_grid - i)
    

@cuda.jit
def velocity_term_kernel(f_x_p_t, velocity_term, masses, \
                         total_grid, num_of_particle_types, \
                         px_grid_size, py_grid_size, pz_grid_size, \
                         x_grid_size, y_grid_size, z_grid_size, \
                         px_left_bound, py_left_bound, pz_left_bound, \
                         dt, dx, dy, dz, dpx, dpy, dpz,\
                         boundary_info_x, boundary_info_y, boundary_info_z, i_time_step):
    
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
            
            # vx, vy, vz
            vx = px/Ep
            vy = py/Ep
            vz = pz/Ep

            # enforce periodical boundary conditions, ixPlus --> ix+1
            ixPlus = (ix+1)%x_grid_size
            iyPlus = (iy+1)%y_grid_size
            izPlus = (iz+1)%z_grid_size

            # -1%3 should be 2, but cuda yields 0, so we use
            ixMinus = (ix+x_grid_size-1)%x_grid_size
            iyMinus = (iy+y_grid_size-1)%y_grid_size
            izMinus = (iz+z_grid_size-1)%z_grid_size

            # distribution functions at x-dx, x, x+dx
            fcurrent = f_x_p_t[p_type, ix, iy, iz, ipx, ipy, ipz]
            fleftx = f_x_p_t[p_type, ixMinus, iy, iz, ipx, ipy, ipz]
            frightx = f_x_p_t[p_type, ixPlus, iy, iz, ipx, ipy, ipz]
            flefty = f_x_p_t[p_type, ix, iyMinus, iz, ipx, ipy, ipz]
            frighty = f_x_p_t[p_type, ix, iyPlus, iz, ipx, ipy, ipz]
            fleftz = f_x_p_t[p_type, ix, iy, izMinus, ipx, ipy, ipz]
            frightz = f_x_p_t[p_type, ix, iy, izPlus, ipx, ipy, ipz]
                               
            # velocity term
            # p/E(x).grad(f,x)
            # find the gradient
            if vx > 0:
                gradx = vx*(fcurrent - fleftx)/dx
            else:
                gradx = vx*(frightx - fcurrent)/dx
        
            if vy > 0:
                grady = vy*(fcurrent - flefty)/dy
            else:
                grady = vy*(frighty - fcurrent)/dy
                
            if vz > 0:
                gradz = vz*(fcurrent - fleftz)/dz
            else:
                gradz = vz*(frightz - fcurrent)/dz
                
            # updattion within the boundary
            if ix > (boundary_info_x[p_type,iy, iz,0])-0.5 and ix < (boundary_info_x[p_type,iy, iz,1])+0.5:
                cuda.atomic.add(velocity_term, (p_type,ix,iy,iz,ipx,ipy,ipz), gradx)
            if boundary_info_x[p_type,iy, iz,0]!=boundary_info_x[p_type,iy, iz,1]:
                # for the left boundary
                if ix == boundary_info_x[p_type,iy, iz,0] - 1:
                    cuda.atomic.add(velocity_term, (p_type,boundary_info_x[p_type,iy, iz,0],iy,iz,ipx,ipy,ipz), gradx)
                # for the right boundary
                if ix == boundary_info_x[p_type,iy, iz,1] + 1:
                    cuda.atomic.add(velocity_term, (p_type,boundary_info_x[p_type,iy, iz,1],iy,iz,ipx,ipy,ipz), gradx)
                
            # updattion within the boundary
            if iy > (boundary_info_y[p_type,iz, ix,0])-0.5 and iy < (boundary_info_y[p_type,iz, ix,1])+0.5:
                cuda.atomic.add(velocity_term, (p_type,ix,iy,iz,ipx,ipy,ipz), grady)
            if boundary_info_y[p_type,iz, ix,0]!=boundary_info_y[p_type,iz, ix,1]:
                # if it is outside the left boundary back propagate
                if iy == boundary_info_y[p_type,iz, ix,0] - 1:
                    cuda.atomic.add(velocity_term, (p_type,ix,boundary_info_y[p_type,iz, ix,0],iz,ipx,ipy,ipz), grady)
                # if it is out side the right boundary back propagate
                if iy == boundary_info_y[p_type,iz, ix,1] + 1:
                    cuda.atomic.add(velocity_term, (p_type,ix,boundary_info_y[p_type,iz, ix,1],iz,ipx,ipy,ipz), grady)
                
            # updattion within the boundary
            if iz > (boundary_info_z[p_type,ix, iy,0])-0.5 and iz < (boundary_info_z[p_type,ix, iy,1])+0.5:
                cuda.atomic.add(velocity_term, (p_type,ix,iy,iz,ipx,ipy,ipz), gradz)
            if boundary_info_z[p_type,ix, iy,0]!=boundary_info_z[p_type,ix, iy,1]:
                # if it is outside the left boundary back propagate
                if iz == boundary_info_z[p_type,ix, iy,0] - 1:
                    cuda.atomic.add(velocity_term, (p_type,ix,iy,boundary_info_z[p_type,ix, iy,0],ipx,ipy,ipz), gradz)
                # if it is out side the right boundary back propagate
                if iz == boundary_info_z[p_type,ix, iy,1] + 1:
                    cuda.atomic.add(velocity_term, (p_type,ix,iy,boundary_info_z[p_type,ix, iy,1],ipx,ipy,ipz), gradz)
        
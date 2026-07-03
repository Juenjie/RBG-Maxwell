from numba import cuda
import math

'''
CUDA kernel for obtaining rho, J densities from given distributions, masses, charges.
This function is purely an integration involving the distributions.

Params
======
f_x_p_t:
    distribution of the particles, of shape [particle_species, nx*ny*nz*npx*npy*npz]
rho, J:
    electric charge and current density at each spatial grid, rho is of size [num_of_particle_types, nx*ny*nz]
masses:
    masses for each type of particles, e.g., [0.2, 0.2]
charges:
    charges for each type of particles, e.g., [0.2, 0.2]
total_grid:
    total number of phase grids, nx*ny*nz*npx*npy*npz
num_of_particle_types:
    total number of particle species, this is len(masses)
npx, npy, npz: 
    number of grid sizes in in momentum domain, e.g., [5,5,5]
nx, ny, nz:
    number of grids in spatial domain, e.g., [5, 5, 5]
half_px, half_py, half_pz:
    left boundary of the momentum region, 
    this is an array containing the values for different type of particles    
dx, dy, dz: 
    infinitesimal difference in spatial coordinate, these are in Energy unit     
dpx, dpy, dpz:
    infinitesimal difference in momentum coordinate, these are in Energy unit
    this is an array containing the values for different type of particles
momentum_space_volume_element:
    dpz*dpy*dpz/(2*math.pi)**3
    this is an array containing the values for different type of particles
total_spatial_grids:
    total number of spatial grids, nx*ny*nz
dp:
    dpx*xpy*dpz
num_momentum_levels:
    number of momentum levels for the distribution
'''

# cuda kernel for obtianing the current densities
@cuda.jit
def rho_J_kernel(f_x_p_t, rho, Jx, Jy, Jz, masses, charges,\
                 total_grid, num_of_particle_types, total_spatial_grids,\
                 momentum_volume_element, npx, npy, npz, \
                 nx, ny, nz, num_momentum_levels,\
                 half_px, half_py, half_pz, \
                 dx, dy, dz, dpx, dpy, dpz, c):
    
    # threads loop in one dimension
    i_grid = cuda.grid(1)
            
    if i_grid < total_grid:
        
        # convert one-d index into six-d
        ipz = i_grid%npz
        ipz_rest = i_grid//npz
        ipy = ipz_rest%npy
        ipy_rest = ipz_rest//npy
        ipx = ipy_rest%npx
        ipx_rest = ipy_rest//npx
        iz = ipx_rest%nz
        iz_rest = ipx_rest//nz
        iy = iz_rest%ny
        iy_rest = iz_rest//ny
        ix = iy_rest%nx
        i_s = iz + iy*nz + ix*nz*ny
        
        # loop through all species
        for p_type in range(num_of_particle_types):
            # distribution function multiplied by volume
            modified_f = f_x_p_t[0, p_type, i_grid]*momentum_volume_element[p_type]

            # if modified_f is less than the numerical accuracy, turn it into 0
            if abs(modified_f) < 10**(-19):
                modified_f = 0.

            # rho at each phase space point
            # rho only considers the 0-th momentum level
            cuda.atomic.add(rho, (p_type, i_s), modified_f)


            # acquire p from the central value
            # Note that for different particles, they have different dpx and px_left_bound
            # the momentum level corresponds to the leel of straitification
            px = ((ipx+0.5)*dpx[p_type] - half_px[p_type])
            py = ((ipy+0.5)*dpy[p_type] - half_py[p_type])
            pz = ((ipz+0.5)*dpz[p_type] - half_pz[p_type])          

            # energy for current grid
            mp_squared = masses[p_type]**2

            # p0 for current grid
            p0 = math.sqrt(mp_squared*c**2+px**2+py**2+pz**2)

            # vx, vy, vz
            vx = 0.
            vy = 0.
            vz = 0.
            if p0 > 10**(-19):
                vx = c*px/p0
                vy = c*py/p0
                vz = c*pz/p0

            # if momentum is in the center grid, the force must be zero
            # here we require that momentum grid must be odd number
            if abs(px) < 0.5*dpx[p_type]:
                vx = 0.
            if abs(py) < 0.5*dpy[p_type]:
                vy = 0.
            if abs(pz) < 0.5*dpz[p_type]:
                vz = 0.  

            fvx = modified_f*vx
            fvy = modified_f*vy
            fvz = modified_f*vz

            # if fv is less than the numerical accuracy, turn it into 0
            if abs(fvx) < 10**(-19):
                fvx = 0.
            if abs(fvy) < 10**(-19):
                fvy = 0.
            if abs(fvz) < 10**(-19):
                fvz = 0.

            # since in the middle of the momentum grid, velocity is zero
            # we need to add the contribution from the higher levels
            cuda.atomic.add(Jx, (p_type, i_s), fvx)
            cuda.atomic.add(Jy, (p_type, i_s), fvy)
            cuda.atomic.add(Jz, (p_type, i_s), fvz)

            
'''
CUDA kernel for obtaining rho (charge), J (electric current) densities from given distributions, masses, charges.
This function is purely an integration involving the distributions.

Params
======
f_x_p_t:
    distribution of the particles, of shape [particle_species, nx*ny*nz*npx*npy*npz]
rho, J:
    electric charge and current density at each spatial grid, rho is of size [num_of_particle_types, nx*ny*nz]
masses:
    masses for each type of particles, e.g., [0.2, 0.2]
charges:
    charges for each type of particles, e.g., [0.2, 0.2]
total_grid:
    total number of phase grids, nx*ny*nz*npx*npy*npz
num_of_particle_types:
    total number of particle species, this is len(masses)
npx, npy, npz: 
    number of grid sizes in in momentum domain, e.g., [5,5,5]
nx, ny, nz:
    number of grids in spatial domain, e.g., [5, 5, 5]
half_px, half_py, half_pz:
    left boundary of the momentum region, 
    this is an array containing the values for different type of particles    
dx, dy, dz: 
    infinitesimal difference in spatial coordinate, these are in Energy unit     
dpx, dpy, dpz:
    infinitesimal difference in momentum coordinate, these are in Energy unit
    this is an array containing the values for different type of particles
momentum_space_volume_element:
    dpz*dpy*dpz/(2*math.pi)**3
    this is an array containing the values for different type of particles
total_spatial_grids:
    total number of spatial grids, nx*ny*nz
dp:
    dpx*xpy*dpz
'''
            
# cuda kernel for obtianing the electric and electric current densities            
@cuda.jit
def electric_rho_J_kernel(f_x_p_t, rho, Jx, Jy, Jz, masses, charges,\
                          total_grid, num_of_particle_types, total_spatial_grids,\
                          momentum_volume_element, npx, npy, npz, nx, ny, nz, \
                          half_px, half_py, half_pz, \
                          dx, dy, dz, dpx, dpy, dpz, num_momentum_levels, c):
    
    # threads loop in one dimension
    i_grid = cuda.grid(1)
            
    if i_grid < total_grid:
        
        # convert one-d index into six-d
        ipz = i_grid%npz
        ipz_rest = i_grid//npz
        ipy = ipz_rest%npy
        ipy_rest = ipz_rest//npy
        ipx = ipy_rest%npx
        ipx_rest = ipy_rest//npx
        iz = ipx_rest%nz
        iz_rest = ipx_rest//nz
        iy = iz_rest%ny
        iy_rest = iz_rest//ny
        ix = iy_rest%nx
        i_s = iz + iy*nz + ix*nz*ny
        
        # loop through all species
        for p_type in range(num_of_particle_types):
            # distribution function multiplied by volume
            modified_f = f_x_p_t[0, p_type, i_grid]*momentum_volume_element[p_type]*charges[p_type]

            # if modified_f is less than the numerical accuracy, turn it into 0
            if abs(modified_f) < 10**(-19):
                modified_f = 0.

            # rho at each phase space point
            # rho only considers the 0-th momentum level
            cuda.atomic.add(rho, i_s, modified_f)

            # acquire p from the central value
            # Note that for different particles, they have different dpx and px_left_bound
            # the momentum level corresponds to the leel of straitification
            px = ((ipx+0.5)*dpx[p_type] - half_px[p_type])
            py = ((ipy+0.5)*dpy[p_type] - half_py[p_type])
            pz = ((ipz+0.5)*dpz[p_type] - half_pz[p_type])            

            # # here we perform restrictions on momentum
            # # if it is the central momentum, turn it into 0
            # if abs(npx/2-ipx)<0.01 and abs(px)<dpx[p_type]*0.1:
            #     px=0.
            # if abs(npy/2-ipy)<0.01 and abs(py)<dpy[p_type]*0.1:
            #     py=0.
            # if abs(npz/2-ipz)<0.01 and abs(pz)<dpz[p_type]*0.1:
            #     pz=0.

            # energy for current grid
            mp_squared = masses[p_type]**2

            # p0 for current grid
            p0 = math.sqrt(mp_squared*c**2+px**2+py**2+pz**2)

            # vx, vy, vz
            vx = 0.
            vy = 0.
            vz = 0.
            if p0 > 10**(-19):
                vx = c*px/p0
                vy = c*py/p0
                vz = c*pz/p0

            # if momentum is in the center grid, the force must be zero
                # here we require that momentum grid must be odd number
            if abs(px) < 0.5*dpx[p_type]:
                vx = 0.
            if abs(py) < 0.5*dpy[p_type]:
                vy = 0.
            if abs(pz) < 0.5*dpz[p_type]:
                vz = 0.  

            fvx = modified_f*vx
            fvy = modified_f*vy
            fvz = modified_f*vz

            # if fv is less than the numerical accuracy, turn it into 0
            if abs(fvx) < 10**(-19):
                fvx = 0.
            if abs(fvy) < 10**(-19):
                fvy = 0.
            if abs(fvz) < 10**(-19):
                fvz = 0.

            cuda.atomic.add(Jx, i_s, fvx)
            cuda.atomic.add(Jy, i_s, fvy)
            cuda.atomic.add(Jz, i_s, fvz)
            
#             if ix==3 and iy==4 and iz==2 and ipy==5 and ipz==6:
#                 print(1000, vx, vy, vz, f_x_p_t[0, p_type, i_grid], momentum_volume_element[p_type], charges[p_type], i_s, Jx[i_s], fvx*10**10, modified_f*10**10)

#             if ix==nx-4 and iy==4 and iz==2 and ipy==5 and ipz==6:
#                 print(2000, vx, vy, vz, f_x_p_t[0, p_type, i_grid], momentum_volume_element[p_type], charges[p_type], i_s, Jx[i_s], fvx*10**10, modified_f*10**10)
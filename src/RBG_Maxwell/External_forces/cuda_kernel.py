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
    masses for each type of particles, e.g., [0.2, 0.2] Energy
charges:
    charges for each type of particles, e.g., [0.2, 0.2]
total_grid:
    total number of grids, nx*ny*nz*npx*npy*npz
num_of_particle_types:
    total number of particle types, this is len(masses)
npx, npy, npz: 
    number of grid sizes in momentum domain, e.g., [5,5,5]
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
Ex, Ey, Ez, Bx, By, Bz:
    electric and magnetic field at each spatial point, of size [3, nx, ny, nz, ]
number_momentum_levels: 
    how many momentum levels are used for the particles
'''

@cuda.jit
def EM_force_kernel(Fx, Fy, Fz, masses, charges,\
                    total_grid, num_of_particle_types, \
                    npx, npy, npz, nx, ny, nz, \
                    half_px, half_py, half_pz, \
                    dx, dy, dz, dpx, dpy, dpz, \
                    Ex, Ey, Ez, Bx, By, Bz, number_momentum_levels, c):
    
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
        
        # E and B
        Efx = Ex[i_s]
        Efy = Ey[i_s]
        Efz = Ez[i_s]
        Bfx = Bx[i_s]
        Bfy = By[i_s]
        Bfz = Bz[i_s]
        
        # loop through all species
        for p_type in range(num_of_particle_types):

            mp_squared = masses[p_type]**2
            
            # loop through all momentum levels
            for i_level in range(number_momentum_levels):            
                
                # acquire p from the central value
                # Note that for different particles, they have different dpx and px_left_bound
                # the momentum level corresponds to the level of straitification
                px = ((ipx+0.5)*dpx[p_type] - half_px[p_type])/(npx**i_level)
                py = ((ipy+0.5)*dpy[p_type] - half_py[p_type])/(npy**i_level)
                pz = ((ipz+0.5)*dpz[p_type] - half_pz[p_type])/(npz**i_level)

                # # here we perform restrictions on momentum
                # # if it is the central momentum, turn it into 0
                # if abs(npx/2-ipx)<0.01 and abs(px)<dpx[p_type]*0.1:
                #     px=0.
                # if abs(npy/2-ipy)<0.01 and abs(py)<dpy[p_type]*0.1:
                #     py=0.
                # if abs(npz/2-ipz)<0.01 and abs(pz)<dpz[p_type]*0.1:
                #     pz=0.
                
                # charge for the current particle
                charge  = charges[p_type]

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
                    
#                 if i_grid == 10000:
#                     # print(10000, vy*10**15, charge*(Efx + (vy*Bfz - vz*Bfy))*10**15, charge*(Efx)*10**15, charge*(Efy)*10**15, charge*((vy*Bfz))*10**15, charge*(vz*Bfy)*10**15, py, ipy, px, ipx)
#                     print(20000,vx,vy,vz,c,px,py,pz,p0,p_type,mp_squared)
                
                # # external EM force    
#                 if ix==0 and iy in [124,126] and iz==55 and ipx==0 and ipy==100 and ipz==0:
#                     print(iy,charge,Efy,vz,Bfx,vx,Bfz,charge*(Efy + (vz*Bfx - vx*Bfz)))
                Fx[i_level,p_type,i_grid] = charge*(Efx + (vy*Bfz - vz*Bfy))
                Fy[i_level,p_type,i_grid] = charge*(Efy + (vz*Bfx - vx*Bfz))
                Fz[i_level,p_type,i_grid] = charge*(Efz + (vx*Bfy - vy*Bfx))
                
                # if ix == 3 and iy == 4 and iz == 2 and ipx == 0 and ipy == 5 and ipz == 5:
                #     print(1000, Fx[i_level,p_type,i_grid], charge, Efx, vy, Bfz, vz, Bfy)
                # if ix == nx-4 and iy == 4 and iz == 2 and ipx == npx-1 and ipy == 5 and ipz == 5:
                #     print(2000, Fx[i_level,p_type,i_grid], charge, Efx, vy, Bfz, vz, Bfy)
                
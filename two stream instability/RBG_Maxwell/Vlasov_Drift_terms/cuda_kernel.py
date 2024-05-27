from numba import cuda
import math

'''
CUDA kernel for obtaining the velocity term using upwind difference.
The device function is to take the upwind difference according to the values of v.
Peoriodical boundary conditions are used for spatial distributions. 

f_x_p_t:
    distribution function for all particles at all levels. The shape of f corresponds to
    [momentum_levels, particle_species, nx*ny*nz*npx*npy*npz].
velocity_term:
    velocity term at each phase space point, of the same shape as f_x_p_t
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
    1. indicates absorption and 0. indicates relection
c:
    the value of speed of light in FU (flexible unit)
'''

@cuda.jit(device=True)
def threeD_to_oneD(ix, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz):
    return ipz+ipy*npz+ipx*npz*npy+iz*npz*npy*npx+iy*npz*npy*npx*nz+ix*npz*npy*npx*nz*ny

@cuda.jit(device=True)
def mirror(ngrid, i):
    return ngrid-i-1
    
@cuda.jit(device=True)
def left_bound_detect(ixMinus, f_x_p_t, i_level, p_type, i_phasexmin):
    # if ixMinus is smaller than 0, set f -> 0
    if ixMinus < -0.5:
        return 0.
    else: 
        return f_x_p_t[i_level, p_type, i_phasexmin]
    
@cuda.jit(device=True)
def right_bound_detect(ixPlus, nx, f_x_p_t, i_level, p_type, i_phasexmax):
    # if ixPlus is larger than nx-1, set f -> 0
    if ixPlus > (nx-0.5):
        return 0.
    else:
        return f_x_p_t[i_level, p_type, i_phasexmax]
    
@cuda.jit(device=True)
def UPFD(dx, vx, fleft, fright):
    # method from doi:10.1016/j.mcm.2011.05.005
    if vx > 0.:
        if abs(vx)<10**(-19):
            vx = 0.
        return vx/dx*fleft
    else:
        if abs(vx)<10**(-19):
            vx = 0.
        return -vx/dx*fright

'''
# gives wrong result, hence decrapted
@cuda.jit(device=True)
def r_minus_i_plus(fleftleft, fleft, fcurrent, fright, frightright):
    if abs((fright-fcurrent))<10**(-13):
        return 0.
    else:
        return (fcurrent-fleft)/(fright-fcurrent)
     
@cuda.jit(device=True)
def r_minus_i_minus(fleftleft, fleft, fcurrent, fright, frightright):
    if abs((fcurrent-fleft))<10**(-13):
        return 0.
    else:
        return (fleft-fleftleft)/(fcurrent-fleft)
    
@cuda.jit(device=True)
def r_plus_i_plus(fleftleft, fleft, fcurrent, fright, frightright):
    if abs((fright-fcurrent))<10**(-13):
        return 0.
    else:
        return (frightright-fright)/(fright-fcurrent)
    
@cuda.jit(device=True)
def r_plus_i_minus(fleftleft, fleft, fcurrent, fright, frightright):
    if abs((fcurrent-fleft))<10**(-13):
        return 0.
    else:
        return (fright-fcurrent)/(fcurrent-fleft)
        
# @cuda.jit(device=True)
# def UPFD_second(dt, dx, vx, fleft, fcurrent, fright, r_minus_i_plus, r_minus_i_minus, r_plus_i_plus, r_plus_i_minus, limiter_type):
#     # method from 
#     if vx > 0.:
#         psi_plus = 1.#psi(r_minus_i_plus, limiter_type)
#         psi_minus = 1.#psi(r_minus_i_minus, limiter_type)
#         return  vx/dx*fleft  + 0.5*(vx)**2*dt/dx**2*(fright+fleft) - min(0.5*abs(vx)*dt/dx*(psi_plus*(fright-fcurrent)-psi_minus*(fcurrent-fleft)), 0.)
#     else:
#         psi_plus = 1.#psi(r_plus_i_plus, limiter_type)
#         psi_minus = 1.#psi(r_plus_i_minus, limiter_type)
#         return -vx/dx*fright + 0.5*(vx)**2*dt/dx**2*(fright+fleft) - min(0.5*abs(vx)*dt/dx*(psi_plus*(fright-fcurrent)-psi_minus*(fcurrent-fleft)), 0.)
'''

'''
# gives wrong result, hence decrapted
# @cuda.jit(device=True)
# def UPFD_second_order_Lax_Wendroff(dt, dx, vx, fleft, fcurrent, fright):
#     return -0.5*vx*dt/dx*(fright-fleft) + 0.5*(vx*dt/dx)**2*(fright-2*fcurrent+fleft)
'''

'''
# gives wrong result, hence decrapted
# @cuda.jit(device=True)
# def UPFD_second_order_USMC(dt, dx, vx, fleft, fcurrent, fright):
#     second_order_term = 0.5*(vx*dt/dx)**2*(fright+fleft)
#     if vx > 0.:
#         return second_order_term + vx*dt/dx*fleft
#     else:
#         return second_order_term - vx*dt/dx*fright
'''

'''
# gives wrong result, hence decrapted
# @cuda.jit(device=True)
# def UPFD_second_order_upwind(dt, dx, vx, fleftleft, fleft, fcurrent, fright, frightright):
#     # method from 
#     if vx > 0.:
#         return -vx*dt/(2*dx)*(3*fcurrent-4*fleft+fleftleft)
#     else:
#         return -vx*dt/(2*dx)*(-frightright+4*fright-3*fcurrent)
'''
'''
# @cuda.jit(device=True)
# def r_minus_i_plus(fleftleft, fleft, fcurrent, fright, frightright):
#     if abs(fcurrent-fleft)<10**(-13):
#         return 0.
#     else:
#         return (fright-fleft)/(fcurrent-fleft)
     
# @cuda.jit(device=True)
# def r_minus_i_minus(fleftleft, fleft, fcurrent, fright, frightright):
#     if abs(fleft-fleftleft)<10**(-13):
#         return 0.
#     else:
#         return (fcurrent-fleft)/(fleft-fleftleft)
    
# @cuda.jit(device=True)
# def r_plus_i_plus(fleftleft, fleft, fcurrent, fright, frightright):
#     if abs(frightright-fright)<10**(-13):
#         return 0.
#     else:
#         return (fright-fcurrent)/(frightright-fright)
    
# @cuda.jit(device=True)
# def r_plus_i_minus(fleftleft, fleft, fcurrent, fright, frightright):
#     if abs(fright-fcurrent)<10**(-13):
#         return 0.
#     else:
#         return (fcurrent-fleft)/(fright-fcurrent)

# @cuda.jit(device=True)
# def superbee(r):
#     return max(0,min(2*r,1),min(r,2))
    
# @cuda.jit(device=True)
# def minmod(r):
#     return max(0, min(1,r))
    
# @cuda.jit(device=True)
# def psi(r, limiter_type):
#     if limiter_type > 0.5:
#         # use superbee
#         return superbee(r)
#     else:
#         # use minmod
#         return minmod(r)
    
# @cuda.jit(device=True)
# def UPFD_second_order_upwind_with_limiter(dt, dx, vx, fleftleft, fleft, fcurrent, fright, frightright, r_minus_i_plus, r_minus_i_minus, r_plus_i_plus, r_plus_i_minus, limiter_type):
#     # method from 
#     if vx > 0.:
#         psi_plus = 1.#psi(r_minus_i_plus, limiter_type)
#         psi_minus = 1.#psi(r_minus_i_minus, limiter_type)
#         return  vx*dt/dx*fleft  - min(0., 0.5*vx*dt/dx*(psi_plus*(fcurrent-fleft)-psi_minus*(fleft-fleftleft)))
#     else:
#         psi_plus = 1.#psi(r_plus_i_plus, limiter_type)
#         psi_minus = 1.#psi(r_plus_i_minus, limiter_type)
#         return -vx*dt/dx*fright - min(0., 0.5*vx*dt/dx*(psi_minus*(fright-fcurrent)-psi_plus*(frightright-fright)))
'''

'''
# # gives wrong result, hence decrapted
# @cuda.jit(device=True)
# def UPFD_second_order_upwind_with_limiter(dt, dx, vx, fleftleft, fleft, fcurrent, fright, frightright):
#         # method from 
#     if vx > 0.:
#         psi_plus = 1.#psi(r_minus_i_plus, limiter_type)
#         psi_minus = 1.#psi(r_minus_i_minus, limiter_type)
#         return  vx*dt/dx*fleft  - min(0., 0.5*vx*dt/dx*(psi_plus*(fcurrent-fleft)-psi_minus*(fleft-fleftleft)))
#     else:
#         psi_plus = 1.#psi(r_plus_i_plus, limiter_type)
#         psi_minus = 1.#psi(r_plus_i_minus, limiter_type)
#         return -vx*dt/dx*fright - min(0., 0.5*vx*dt/dx*(psi_minus*(fright-fcurrent)-psi_plus*(frightright-fright)))
'''

#######################################################
# see method in Bernard Parent, Positivity-preserving flux-limited method for compressible fluid flow, 2011
# define flux F+-
@cuda.jit(device=True)
def F_plus_minus(sign, vx, fcurrent):
    if vx*sign > 0.:
        return vx*fcurrent
    else:
        return 0.

# define flux F_i_plus_half
@cuda.jit(device=True)
def F_i_plus_half(vx, F_plus_i, phi_plus_i_plus_half, F_plus_i_minus_1, F_minus_i_plus_1, phi_minus_i_plus_half, F_minus_i_plus_2):
    return F_plus_i + 0.5*phi_plus_i_plus_half*(F_plus_i-F_plus_i_minus_1) + F_minus_i_plus_1 + 0.5*phi_minus_i_plus_half*(F_minus_i_plus_1-F_minus_i_plus_2)
    
# define limiter function phi_minus
@cuda.jit(device=True)
def phi_minus(F_minus, F_minus_right, F_minus_rightright, theta):
    if abs(F_minus_right-F_minus_rightright) < 10**(-18):
        monotonicity_preservation = 0.
    else:
        monotonicity_preservation = (F_minus-F_minus_right)/(F_minus_right-F_minus_rightright)
    if abs(F_minus_right) < 10**(-18):
        posivity_preservation = 2/max(theta, 0.) 
    else:
        posivity_preservation = 2/max(theta, (F_minus_rightright-F_minus_right)/F_minus_right) 
    return max(0., min(1., monotonicity_preservation, posivity_preservation))
 
# define limiter function phi_plus
@cuda.jit(device=True)
def phi_plus(F_plus_left, F_plus, F_plus_right, theta):
    if abs(F_plus-F_plus_left) < 10**(-18):
        monotonicity_preservation = 0.
    else:
        monotonicity_preservation = (F_plus_right-F_plus)/(F_plus-F_plus_left)
    if abs(F_plus) < 10**(-18):
        posivity_preservation = 2/max(theta, 0.)    
    else:
        posivity_preservation = 2/max(theta, (F_plus_left-F_plus)/F_plus)    
    return max(0., min(1., monotonicity_preservation, posivity_preservation))
#######################################################

@cuda.jit(device=True)
def first_order_upwind(dt, dx, vx, fleft, fcurrent, fright):
    if vx > 0.:
        return (fcurrent-fleft)/dx*vx*dt
    else:
        return (fright-fcurrent)/dx*vx*dt

@cuda.jit
def drift_force_term_kernel(f_x_p_t, Fx, Fy, Fz, \
                            masses, total_grid, num_of_particle_types, \
                            npx, npy, npz, nx, ny, nz,\
                            half_px, half_py, half_pz, \
                            dx, dy, dz, dpx, dpy, dpz, number_momentum_levels,\
                            x_bound_config, y_bound_config, z_bound_config, \
                            f_updated, collision_term, \
                            dt, c, current_time_step, whether_drift, whether_Vlasov, \
                            drift_order):

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
# 
        if True:

            # distribution functions out of the computation domain of each card are always set to be 0
    
            ixPlus = (ix+1) %nx
            iyPlus = (iy+1) %ny
            izPlus = (iz+1) %nz
            ixMinus = (ix-1) %nx
            iyMinus = (iy-1) %ny
            izMinus = (iz-1) %nz
            ixPlusPlus = (ix+2+nx) %nx
            iyPlusPlus = (iy+2+ny) %ny
            izPlusPlus = (iz+2+nz) %nz
            ixMinusMinus = (ix-2+nx) %nx
            iyMinusMinus = (iy-2+ny) %ny
            izMinusMinus = (iz-2+nz) %nz

            # convert six-d to one-d 
            i_phasexmin = threeD_to_oneD(ixMinus, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasexmax = threeD_to_oneD(ixPlus, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phaseymin = threeD_to_oneD(ix, iyMinus, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phaseymax = threeD_to_oneD(ix, iyPlus, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasezmin = threeD_to_oneD(ix, iy, izMinus, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasezmax = threeD_to_oneD(ix, iy, izPlus, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasexminmin = threeD_to_oneD(ixMinusMinus, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasexmaxmax = threeD_to_oneD(ixPlusPlus, iy, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phaseyminmin = threeD_to_oneD(ix, iyMinusMinus, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phaseymaxmax = threeD_to_oneD(ix, iyPlusPlus, iz, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasezminmin = threeD_to_oneD(ix, iy, izMinusMinus, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasezmaxmax = threeD_to_oneD(ix, iy, izPlusPlus, ipx, ipy, ipz, nx, ny, nz, npx, npy, npz)

            # enforce 
            ipxPlus = (ipx+1)
            ipyPlus = (ipy+1)
            ipzPlus = (ipz+1)
            ipxPlusPlus = (ipx+2)
            ipyPlusPlus = (ipy+2)
            ipzPlusPlus = (ipz+2)

            # 
            ipxMinus = (ipx-1)
            ipyMinus = (ipy-1)
            ipzMinus = (ipz-1)
            ipxMinusMinus = (ipx-2)
            ipyMinusMinus = (ipy-2)
            ipzMinusMinus = (ipz-2)

            # convert six-d to one-d 
            i_phasepxmin = threeD_to_oneD(ix, iy, iz, ipxMinus, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepxmax = threeD_to_oneD(ix, iy, iz, ipxPlus, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepymin = threeD_to_oneD(ix, iy, iz, ipx, ipyMinus, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepymax = threeD_to_oneD(ix, iy, iz, ipx, ipyPlus, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepzmin = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipzMinus, nx, ny, nz, npx, npy, npz)
            i_phasepzmax = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipzPlus, nx, ny, nz, npx, npy, npz)
            i_phasepxminmin = threeD_to_oneD(ix, iy, iz, ipxMinusMinus, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepxmaxmax = threeD_to_oneD(ix, iy, iz, ipxPlusPlus, ipy, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepyminmin = threeD_to_oneD(ix, iy, iz, ipx, ipyMinusMinus, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepymaxmax = threeD_to_oneD(ix, iy, iz, ipx, ipyPlusPlus, ipz, nx, ny, nz, npx, npy, npz)
            i_phasepzminmin = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipzMinusMinus, nx, ny, nz, npx, npy, npz)
            i_phasepzmaxmax = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipzPlusPlus, nx, ny, nz, npx, npy, npz)

            # loop through all species
            # only update the electrons
            for p_type in range(1):    

                # masses
                mp_squared = masses[p_type]**2

                # loop through all momentum levels
                for i_level in range(number_momentum_levels):

                    fcurrent = f_x_p_t[i_level, p_type, i_grid] 

                    # parts of the numerator in UPFD
                    numerator0 = fcurrent/dt/3.

                    # vx, vy, vz
                    vx = 0.
                    vy = 0.
                    vz = 0.
                    # Fx, Fy, Fz
                    Fcurrentpx = 0.
                    Fcurrentpy = 0.
                    Fcurrentpz = 0.               

                    # acquire p from the central value
                    # Note that for different particles, they have different dpx and px_left_bound
                    # the momentum level corresponds to the level of straitification
                    px = ((ipx+0.5)*dpx[p_type] - half_px[p_type])/(npx**i_level)
                    py = ((ipy+0.5)*dpy[p_type] - half_py[p_type])/(npy**i_level)
                    pz = ((ipz+0.5)*dpz[p_type] - half_pz[p_type])/(npz**i_level)

                    # p0 for current grid
                    p0 = math.sqrt(mp_squared*c**2+px**2+py**2+pz**2)
                    #  p0 = math.sqrt((mp_squared*c)**2+px**2+py**2+pz**2)
                    if p0 > 10**(-13):
                        vx = c*px/p0
                        vy = c*py/p0
                        vz = c*pz/p0

                    # distribution functions at x-dx, x, x+dx               
                    fleftx = left_bound_detect(ixMinus, f_x_p_t, i_level, p_type, i_phasexmin)
                    flefty = left_bound_detect(iyMinus, f_x_p_t, i_level, p_type, i_phaseymin)
                    fleftz = left_bound_detect(izMinus, f_x_p_t, i_level, p_type, i_phasezmin)
                    frightx = right_bound_detect(ixPlus, nx, f_x_p_t, i_level, p_type, i_phasexmax)
                    frighty = right_bound_detect(iyPlus, ny, f_x_p_t, i_level, p_type, i_phaseymax)
                    frightz = right_bound_detect(izPlus, nz, f_x_p_t, i_level, p_type, i_phasezmax)
                    fleftleftx = left_bound_detect(ixMinusMinus, f_x_p_t, i_level, p_type, i_phasexminmin)
                    fleftlefty = left_bound_detect(iyMinusMinus, f_x_p_t, i_level, p_type, i_phaseyminmin)
                    fleftleftz = left_bound_detect(izMinusMinus, f_x_p_t, i_level, p_type, i_phasezminmin)
                    frightrightx = right_bound_detect(ixPlusPlus, nx, f_x_p_t, i_level, p_type, i_phasexmaxmax)
                    frightrighty = right_bound_detect(iyPlusPlus, ny, f_x_p_t, i_level, p_type, i_phaseymaxmax)
                    frightrightz = right_bound_detect(izPlusPlus, nz, f_x_p_t, i_level, p_type, i_phasezmaxmax)
                    
                    
                    # distribution functions at px-dpx, px, px+dpx 
                    fleftpx = left_bound_detect(ipxMinus, f_x_p_t, i_level, p_type, i_phasepxmin)
                    fleftpy = left_bound_detect(ipyMinus, f_x_p_t, i_level, p_type, i_phasepymin)
                    fleftpz = left_bound_detect(ipzMinus, f_x_p_t, i_level, p_type, i_phasepzmin)
                    frightpx = right_bound_detect(ipxPlus, npx, f_x_p_t, i_level, p_type, i_phasepxmax)
                    frightpy = right_bound_detect(ipyPlus, npy, f_x_p_t, i_level, p_type, i_phasepymax)
                    frightpz = right_bound_detect(ipzPlus, npz, f_x_p_t, i_level, p_type, i_phasepzmax)
                    fleftleftpx = left_bound_detect(ipxMinusMinus, f_x_p_t, i_level, p_type, i_phasepxminmin)
                    fleftleftpy = left_bound_detect(ipyMinusMinus, f_x_p_t, i_level, p_type, i_phasepyminmin)
                    fleftleftpz = left_bound_detect(ipzMinusMinus, f_x_p_t, i_level, p_type, i_phasepzminmin)
                    frightrightpx = right_bound_detect(ipxPlusPlus, npx, f_x_p_t, i_level, p_type, i_phasepxmaxmax)
                    frightrightpy = right_bound_detect(ipyPlusPlus, npy, f_x_p_t, i_level, p_type, i_phasepymaxmax)
                    frightrightpz = right_bound_detect(ipzPlusPlus, npz, f_x_p_t, i_level, p_type, i_phasepzmaxmax)

                    # External forces at p, p-dpx, px+dpx
                    Fcurrentpx = Fx[i_level, p_type, i_grid]
                    Fcurrentpy = Fy[i_level, p_type, i_grid]
                    Fcurrentpz = Fz[i_level, p_type, i_grid]
                            
    ############################################################################################################# 
                    if drift_order < 1.5:
                        if whether_drift > 0.5 and whether_Vlasov > 0.5:
                            # first order upwind                            
                            D_sum = abs(vx/dx)+abs(vy/dy)+abs(vz/dz) 
                            V_sum = abs(Fcurrentpx/dpx[p_type])+abs(Fcurrentpy/dpy[p_type])+abs(Fcurrentpz/dpz[p_type])
                            denominator_UPFD = 1 + dt*(D_sum + V_sum)
                            numerator1 = UPFD(dx, vx, fleftx, frightx) + UPFD(dy, vy, flefty, frighty) + UPFD(dz, vz, fleftz, frightz) 
                            numerator2 = UPFD(dpx[p_type], Fcurrentpx, fleftpx, frightpx) + UPFD(dpy[p_type], Fcurrentpy, fleftpy, frightpy) + UPFD(dpz[p_type], Fcurrentpz, fleftpz, frightpz) 
                            f_drift = (fcurrent + dt*collision_term[i_level, p_type, i_grid] + dt*(numerator1 + numerator2))/denominator_UPFD
#                             if current_time_step%1000==0:
#                                 print('f_drift: ', f_drift, 'fcurrent: ', fcurrent, 'dt*collision_term: ', dt*collision_term[i_level, p_type, i_grid], 'dt: ', dt, 'numerator1: ', numerator1, 'numerator2: ', numerator2, 'denominator_UPFD: ', denominator_UPFD, 'D_sum: ', D_sum, 'V_sum: ', V_sum, 'vx: ', vx, 'Fcurrentpx: ', Fcurrentpx, 'dx: ', dx, 'dpx: ', dpx[p_type])
                                
                                
                        elif whether_drift > 0.5 and whether_Vlasov < 0.5:
                            # first order upwind                            
                            D_sum = abs(vx/dx)+abs(vy/dy)+abs(vz/dz)
                            denominator_UPFD = 1 + dt*(D_sum)
                            numerator1 = UPFD(dx, vx, fleftx, frightx) + UPFD(dy, vy, flefty, frighty) + UPFD(dz, vz, fleftz, frightz)
                            f_drift = (fcurrent + dt*collision_term[i_level, p_type, i_grid] + dt*numerator1)/denominator_UPFD
                            
                        elif whether_drift < 0.5 and whether_Vlasov > 0.5:
                            # first order upwind                            
                            V_sum = abs(Fcurrentpx/dpx[p_type])+abs(Fcurrentpy/dpy[p_type])+abs(Fcurrentpz/dpz[p_type])
                            denominator_UPFD = 1 + dt*(V_sum)
                            numerator2 = UPFD(dpx[p_type], Fcurrentpx, fleftpx, frightpx) + UPFD(dpy[p_type], Fcurrentpy, fleftpy, frightpy) + UPFD(dpz[p_type], Fcurrentpz, fleftpz, frightpz) 
                            f_drift = (fcurrent + dt*collision_term[i_level, p_type, i_grid] + dt*numerator2)/denominator_UPFD
                            
                        elif whether_drift < 0.5 and whether_Vlasov < 0.5:
                            f_drift = fcurrent + dt*collision_term[i_level, p_type, i_grid]
    
                        f_updated[i_level, p_type, i_grid] = f_drift 
                        
                    else:
                    
                        if whether_drift > 0.5 and whether_Vlasov > 0.5:
    
                            theta = 10**(-10)

                            ###############################################
                            ###############################################
                            # D_sum
                            ###############################################
                            # flux
                            F_plus_i_x = F_plus_minus(1, vx, fcurrent)
                            F_plus_i_plus_1_x = F_plus_minus(1, vx, frightx)
                            F_plus_i_minus_1_x = F_plus_minus(1, vx, fleftx)
                            F_plus_i_minus_2_x = F_plus_minus(1, vx, fleftleftx)
                            F_minus_i_x = F_plus_minus(-1, vx, fcurrent)  
                            F_minus_i_minus_1_x = F_plus_minus(-1, vx, fleftx)
                            F_minus_i_plus_1_x = F_plus_minus(-1, vx, frightx)
                            F_minus_i_plus_2_x = F_plus_minus(-1, vx, frightrightx)

                            # flux limiter
                            phi_plus_i_plus_half_x = phi_plus(F_plus_i_minus_1_x, F_plus_i_x, F_plus_i_plus_1_x, theta)
                            phi_minus_i_plus_half_x = phi_minus(F_minus_i_x, F_minus_i_plus_1_x, F_minus_i_plus_2_x, theta)
                            phi_plus_i_minus_half_x = phi_plus(F_plus_i_minus_2_x, F_plus_i_minus_1_x, F_plus_i_x, theta)
                            phi_minus_i_minus_half_x = phi_minus(F_minus_i_minus_1_x, F_minus_i_x, F_minus_i_plus_1_x, theta)

                            # flux at the middle grid
                            F_i_plus_half_x = F_i_plus_half(vx, F_plus_i_x, phi_plus_i_plus_half_x, F_plus_i_minus_1_x, F_minus_i_plus_1_x, phi_minus_i_plus_half_x, F_minus_i_plus_2_x)
                            F_i_minus_half_x = F_i_plus_half(vx, F_plus_i_minus_1_x, phi_plus_i_minus_half_x, F_plus_i_minus_2_x, F_minus_i_x, phi_minus_i_minus_half_x, F_minus_i_plus_1_x)
                            #################################################
                            # flux
                            F_plus_i_y = F_plus_minus(1, vy, fcurrent)
                            F_plus_i_plus_1_y = F_plus_minus(1, vy, frighty)
                            F_plus_i_minus_1_y = F_plus_minus(1, vy, flefty)
                            F_plus_i_minus_2_y = F_plus_minus(1, vy, fleftlefty)
                            F_minus_i_y = F_plus_minus(-1, vy, fcurrent)  
                            F_minus_i_minus_1_y = F_plus_minus(-1, vy, flefty)
                            F_minus_i_plus_1_y = F_plus_minus(-1, vy, frighty)
                            F_minus_i_plus_2_y = F_plus_minus(-1, vy, frightrighty)

                            # flux limiter
                            phi_plus_i_plus_half_y = phi_plus(F_plus_i_minus_1_y, F_plus_i_y, F_plus_i_plus_1_y, theta)
                            phi_minus_i_plus_half_y = phi_minus(F_minus_i_y, F_minus_i_plus_1_y, F_minus_i_plus_2_y, theta)
                            phi_plus_i_minus_half_y = phi_plus(F_plus_i_minus_2_y, F_plus_i_minus_1_y, F_plus_i_y, theta)
                            phi_minus_i_minus_half_y = phi_minus(F_minus_i_minus_1_y, F_minus_i_y, F_minus_i_plus_1_y, theta)

                            # flux at the middle grid
                            F_i_plus_half_y = F_i_plus_half(vy, F_plus_i_y, phi_plus_i_plus_half_y, F_plus_i_minus_1_y, F_minus_i_plus_1_y, phi_minus_i_plus_half_y, F_minus_i_plus_2_y)
                            F_i_minus_half_y = F_i_plus_half(vy, F_plus_i_minus_1_y, phi_plus_i_minus_half_y, F_plus_i_minus_2_y, F_minus_i_y, phi_minus_i_minus_half_y, F_minus_i_plus_1_y)

                            ####################################################
                            # flux
                            F_plus_i_z = F_plus_minus(1, vz, fcurrent)
                            F_plus_i_plus_1_z = F_plus_minus(1, vz, frightz)
                            F_plus_i_minus_1_z = F_plus_minus(1, vz, fleftz)
                            F_plus_i_minus_2_z = F_plus_minus(1, vz, fleftleftz)
                            F_minus_i_z = F_plus_minus(-1, vz, fcurrent)  
                            F_minus_i_minus_1_z = F_plus_minus(-1, vz, fleftz)
                            F_minus_i_plus_1_z = F_plus_minus(-1, vz, frightz)
                            F_minus_i_plus_2_z = F_plus_minus(-1, vz, frightrightz)

                            # flux limiter
                            phi_plus_i_plus_half_z = phi_plus(F_plus_i_minus_1_z, F_plus_i_z, F_plus_i_plus_1_z, theta)
                            phi_minus_i_plus_half_z = phi_minus(F_minus_i_z, F_minus_i_plus_1_z, F_minus_i_plus_2_z, theta)
                            phi_plus_i_minus_half_z = phi_plus(F_plus_i_minus_2_z, F_plus_i_minus_1_z, F_plus_i_z, theta)
                            phi_minus_i_minus_half_z = phi_minus(F_minus_i_minus_1_z, F_minus_i_z, F_minus_i_plus_1_z, theta)

                            # flux at the middle grid
                            F_i_plus_half_z = F_i_plus_half(vz, F_plus_i_z, phi_plus_i_plus_half_z, F_plus_i_minus_1_z, F_minus_i_plus_1_z, phi_minus_i_plus_half_z, F_minus_i_plus_2_z)
                            F_i_minus_half_z = F_i_plus_half(vz, F_plus_i_minus_1_z, phi_plus_i_minus_half_z, F_plus_i_minus_2_z, F_minus_i_z, phi_minus_i_minus_half_z, F_minus_i_plus_1_z)

                            D_sum_second = dt/dx*(F_i_plus_half_x-F_i_minus_half_x) + dt/dy*(F_i_plus_half_y-F_i_minus_half_y) + dt/dz*(F_i_plus_half_z-F_i_minus_half_z)
                            
                            ###############################################
                            ###############################################
                            # V_sum
                            ###############################################
                            # flux
                            F_plus_i_x = F_plus_minus(1, Fcurrentpx, fcurrent)
                            F_plus_i_plus_1_x = F_plus_minus(1, Fcurrentpx, frightpx)
                            F_plus_i_minus_1_x = F_plus_minus(1, Fcurrentpx, fleftpx)
                            F_plus_i_minus_2_x = F_plus_minus(1, Fcurrentpx, fleftleftpx)
                            F_minus_i_x = F_plus_minus(-1, Fcurrentpx, fcurrent)  
                            F_minus_i_minus_1_x = F_plus_minus(-1, Fcurrentpx, fleftpx)
                            F_minus_i_plus_1_x = F_plus_minus(-1, Fcurrentpx, frightpx)
                            F_minus_i_plus_2_x = F_plus_minus(-1, Fcurrentpx, frightrightpx)

                            # flux limiter
                            phi_plus_i_plus_half_x = phi_plus(F_plus_i_minus_1_x, F_plus_i_x, F_plus_i_plus_1_x, theta)
                            phi_minus_i_plus_half_x = phi_minus(F_minus_i_x, F_minus_i_plus_1_x, F_minus_i_plus_2_x, theta)
                            phi_plus_i_minus_half_x = phi_plus(F_plus_i_minus_2_x, F_plus_i_minus_1_x, F_plus_i_x, theta)
                            phi_minus_i_minus_half_x = phi_minus(F_minus_i_minus_1_x, F_minus_i_x, F_minus_i_plus_1_x, theta)

                            # flux at the middle grid
                            F_i_plus_half_x = F_i_plus_half(Fcurrentpx, F_plus_i_x, phi_plus_i_plus_half_x, F_plus_i_minus_1_x, F_minus_i_plus_1_x, phi_minus_i_plus_half_x, F_minus_i_plus_2_x)
                            F_i_minus_half_x = F_i_plus_half(Fcurrentpx, F_plus_i_minus_1_x, phi_plus_i_minus_half_x, F_plus_i_minus_2_x, F_minus_i_x, phi_minus_i_minus_half_x, F_minus_i_plus_1_x)
                            #################################################
                            # flux
                            F_plus_i_y = F_plus_minus(1, Fcurrentpy, fcurrent)
                            F_plus_i_plus_1_y = F_plus_minus(1, Fcurrentpy, frightpy)
                            F_plus_i_minus_1_y = F_plus_minus(1, Fcurrentpy, fleftpy)
                            F_plus_i_minus_2_y = F_plus_minus(1, Fcurrentpy, fleftleftpy)
                            F_minus_i_y = F_plus_minus(-1, Fcurrentpy, fcurrent)  
                            F_minus_i_minus_1_y = F_plus_minus(-1, Fcurrentpy, fleftpy)
                            F_minus_i_plus_1_y = F_plus_minus(-1, Fcurrentpy, frightpy)
                            F_minus_i_plus_2_y = F_plus_minus(-1, Fcurrentpy, frightrightpy)

                            # flux limiter
                            phi_plus_i_plus_half_y = phi_plus(F_plus_i_minus_1_y, F_plus_i_y, F_plus_i_plus_1_y, theta)
                            phi_minus_i_plus_half_y = phi_minus(F_minus_i_y, F_minus_i_plus_1_y, F_minus_i_plus_2_y, theta)
                            phi_plus_i_minus_half_y = phi_plus(F_plus_i_minus_2_y, F_plus_i_minus_1_y, F_plus_i_y, theta)
                            phi_minus_i_minus_half_y = phi_minus(F_minus_i_minus_1_y, F_minus_i_y, F_minus_i_plus_1_y, theta)

                            # flux at the middle grid
                            F_i_plus_half_y = F_i_plus_half(Fcurrentpy, F_plus_i_y, phi_plus_i_plus_half_y, F_plus_i_minus_1_y, F_minus_i_plus_1_y, phi_minus_i_plus_half_y, F_minus_i_plus_2_y)
                            F_i_minus_half_y = F_i_plus_half(Fcurrentpy, F_plus_i_minus_1_y, phi_plus_i_minus_half_y, F_plus_i_minus_2_y, F_minus_i_y, phi_minus_i_minus_half_y, F_minus_i_plus_1_y)

                            ####################################################
                            # flux
                            F_plus_i_z = F_plus_minus(1, Fcurrentpz, fcurrent)
                            F_plus_i_plus_1_z = F_plus_minus(1, Fcurrentpz, frightpz)
                            F_plus_i_minus_1_z = F_plus_minus(1, Fcurrentpz, fleftpz)
                            F_plus_i_minus_2_z = F_plus_minus(1, Fcurrentpz, fleftleftpz)
                            F_minus_i_z = F_plus_minus(-1, Fcurrentpz, fcurrent)  
                            F_minus_i_minus_1_z = F_plus_minus(-1, Fcurrentpz, fleftpz)
                            F_minus_i_plus_1_z = F_plus_minus(-1, Fcurrentpz, frightpz)
                            F_minus_i_plus_2_z = F_plus_minus(-1, Fcurrentpz, frightrightpz)

                            # flux limiter
                            phi_plus_i_plus_half_z = phi_plus(F_plus_i_minus_1_z, F_plus_i_z, F_plus_i_plus_1_z, theta)
                            phi_minus_i_plus_half_z = phi_minus(F_minus_i_z, F_minus_i_plus_1_z, F_minus_i_plus_2_z, theta)
                            phi_plus_i_minus_half_z = phi_plus(F_plus_i_minus_2_z, F_plus_i_minus_1_z, F_plus_i_z, theta)
                            phi_minus_i_minus_half_z = phi_minus(F_minus_i_minus_1_z, F_minus_i_z, F_minus_i_plus_1_z, theta)

                            # flux at the middle grid
                            F_i_plus_half_z = F_i_plus_half(Fcurrentpz, F_plus_i_z, phi_plus_i_plus_half_z, F_plus_i_minus_1_z, F_minus_i_plus_1_z, phi_minus_i_plus_half_z, F_minus_i_plus_2_z)
                            F_i_minus_half_z = F_i_plus_half(Fcurrentpz, F_plus_i_minus_1_z, phi_plus_i_minus_half_z, F_plus_i_minus_2_z, F_minus_i_z, phi_minus_i_minus_half_z, F_minus_i_plus_1_z)

                            V_sum_second = dt/dpx[p_type]*(F_i_plus_half_x-F_i_minus_half_x) + dt/dpy[p_type]*(F_i_plus_half_y-F_i_minus_half_y) + dt/dpz[p_type]*(F_i_plus_half_z-F_i_minus_half_z)
                            
                            ####################################################
                            ####################################################
                            f_drift = fcurrent + dt*collision_term[i_level, p_type, i_grid] - D_sum_second - V_sum_second
                            
                        elif whether_drift > 0.5 and whether_Vlasov < 0.5:
                           
                            theta = 10**(-10)

                            ###############################################
                            ###############################################
                            # D_sum
                            ###############################################
                            # flux
                            F_plus_i_x = F_plus_minus(1, vx, fcurrent)
                            F_plus_i_plus_1_x = F_plus_minus(1, vx, frightx)
                            F_plus_i_minus_1_x = F_plus_minus(1, vx, fleftx)
                            F_plus_i_minus_2_x = F_plus_minus(1, vx, fleftleftx)
                            F_minus_i_x = F_plus_minus(-1, vx, fcurrent)  
                            F_minus_i_minus_1_x = F_plus_minus(-1, vx, fleftx)
                            F_minus_i_plus_1_x = F_plus_minus(-1, vx, frightx)
                            F_minus_i_plus_2_x = F_plus_minus(-1, vx, frightrightx)

                            # flux limiter
                            phi_plus_i_plus_half_x = phi_plus(F_plus_i_minus_1_x, F_plus_i_x, F_plus_i_plus_1_x, theta)
                            phi_minus_i_plus_half_x = phi_minus(F_minus_i_x, F_minus_i_plus_1_x, F_minus_i_plus_2_x, theta)
                            phi_plus_i_minus_half_x = phi_plus(F_plus_i_minus_2_x, F_plus_i_minus_1_x, F_plus_i_x, theta)
                            phi_minus_i_minus_half_x = phi_minus(F_minus_i_minus_1_x, F_minus_i_x, F_minus_i_plus_1_x, theta)

                            # flux at the middle grid
                            F_i_plus_half_x = F_i_plus_half(vx, F_plus_i_x, phi_plus_i_plus_half_x, F_plus_i_minus_1_x, F_minus_i_plus_1_x, phi_minus_i_plus_half_x, F_minus_i_plus_2_x)
                            F_i_minus_half_x = F_i_plus_half(vx, F_plus_i_minus_1_x, phi_plus_i_minus_half_x, F_plus_i_minus_2_x, F_minus_i_x, phi_minus_i_minus_half_x, F_minus_i_plus_1_x)
                            #################################################
                            # flux
                            F_plus_i_y = F_plus_minus(1, vy, fcurrent)
                            F_plus_i_plus_1_y = F_plus_minus(1, vy, frighty)
                            F_plus_i_minus_1_y = F_plus_minus(1, vy, flefty)
                            F_plus_i_minus_2_y = F_plus_minus(1, vy, fleftlefty)
                            F_minus_i_y = F_plus_minus(-1, vy, fcurrent)  
                            F_minus_i_minus_1_y = F_plus_minus(-1, vy, flefty)
                            F_minus_i_plus_1_y = F_plus_minus(-1, vy, frighty)
                            F_minus_i_plus_2_y = F_plus_minus(-1, vy, frightrighty)

                            # flux limiter
                            phi_plus_i_plus_half_y = phi_plus(F_plus_i_minus_1_y, F_plus_i_y, F_plus_i_plus_1_y, theta)
                            phi_minus_i_plus_half_y = phi_minus(F_minus_i_y, F_minus_i_plus_1_y, F_minus_i_plus_2_y, theta)
                            phi_plus_i_minus_half_y = phi_plus(F_plus_i_minus_2_y, F_plus_i_minus_1_y, F_plus_i_y, theta)
                            phi_minus_i_minus_half_y = phi_minus(F_minus_i_minus_1_y, F_minus_i_y, F_minus_i_plus_1_y, theta)

                            # flux at the middle grid
                            F_i_plus_half_y = F_i_plus_half(vy, F_plus_i_y, phi_plus_i_plus_half_y, F_plus_i_minus_1_y, F_minus_i_plus_1_y, phi_minus_i_plus_half_y, F_minus_i_plus_2_y)
                            F_i_minus_half_y = F_i_plus_half(vy, F_plus_i_minus_1_y, phi_plus_i_minus_half_y, F_plus_i_minus_2_y, F_minus_i_y, phi_minus_i_minus_half_y, F_minus_i_plus_1_y)

                            ####################################################
                            # flux
                            F_plus_i_z = F_plus_minus(1, vz, fcurrent)
                            F_plus_i_plus_1_z = F_plus_minus(1, vz, frightz)
                            F_plus_i_minus_1_z = F_plus_minus(1, vz, fleftz)
                            F_plus_i_minus_2_z = F_plus_minus(1, vz, fleftleftz)
                            F_minus_i_z = F_plus_minus(-1, vz, fcurrent)  
                            F_minus_i_minus_1_z = F_plus_minus(-1, vz, fleftz)
                            F_minus_i_plus_1_z = F_plus_minus(-1, vz, frightz)
                            F_minus_i_plus_2_z = F_plus_minus(-1, vz, frightrightz)

                            # flux limiter
                            phi_plus_i_plus_half_z = phi_plus(F_plus_i_minus_1_z, F_plus_i_z, F_plus_i_plus_1_z, theta)
                            phi_minus_i_plus_half_z = phi_minus(F_minus_i_z, F_minus_i_plus_1_z, F_minus_i_plus_2_z, theta)
                            phi_plus_i_minus_half_z = phi_plus(F_plus_i_minus_2_z, F_plus_i_minus_1_z, F_plus_i_z, theta)
                            phi_minus_i_minus_half_z = phi_minus(F_minus_i_minus_1_z, F_minus_i_z, F_minus_i_plus_1_z, theta)

                            # flux at the middle grid
                            F_i_plus_half_z = F_i_plus_half(vz, F_plus_i_z, phi_plus_i_plus_half_z, F_plus_i_minus_1_z, F_minus_i_plus_1_z, phi_minus_i_plus_half_z, F_minus_i_plus_2_z)
                            F_i_minus_half_z = F_i_plus_half(vz, F_plus_i_minus_1_z, phi_plus_i_minus_half_z, F_plus_i_minus_2_z, F_minus_i_z, phi_minus_i_minus_half_z, F_minus_i_plus_1_z)

                            D_sum_second = dt/dx*(F_i_plus_half_x-F_i_minus_half_x) + dt/dy*(F_i_plus_half_y-F_i_minus_half_y) + dt/dz*(F_i_plus_half_z-F_i_minus_half_z)
                            
                            ####################################################
                            ####################################################
                            f_drift = fcurrent + dt*collision_term[i_level, p_type, i_grid] - D_sum_second
                            
                        elif whether_drift < 0.5 and whether_Vlasov > 0.5:
                            
                            theta = 10**(-10)
                           
                            ###############################################
                            ###############################################
                            # V_sum
                            ###############################################
                            # flux
                            F_plus_i_x = F_plus_minus(1, Fcurrentpx, fcurrent)
                            F_plus_i_plus_1_x = F_plus_minus(1, Fcurrentpx, frightpx)
                            F_plus_i_minus_1_x = F_plus_minus(1, Fcurrentpx, fleftpx)
                            F_plus_i_minus_2_x = F_plus_minus(1, Fcurrentpx, fleftleftpx)
                            F_minus_i_x = F_plus_minus(-1, Fcurrentpx, fcurrent)  
                            F_minus_i_minus_1_x = F_plus_minus(-1, Fcurrentpx, fleftpx)
                            F_minus_i_plus_1_x = F_plus_minus(-1, Fcurrentpx, frightpx)
                            F_minus_i_plus_2_x = F_plus_minus(-1, Fcurrentpx, frightrightpx)

                            # flux limiter
                            phi_plus_i_plus_half_x = phi_plus(F_plus_i_minus_1_x, F_plus_i_x, F_plus_i_plus_1_x, theta)
                            phi_minus_i_plus_half_x = phi_minus(F_minus_i_x, F_minus_i_plus_1_x, F_minus_i_plus_2_x, theta)
                            phi_plus_i_minus_half_x = phi_plus(F_plus_i_minus_2_x, F_plus_i_minus_1_x, F_plus_i_x, theta)
                            phi_minus_i_minus_half_x = phi_minus(F_minus_i_minus_1_x, F_minus_i_x, F_minus_i_plus_1_x, theta)

                            # flux at the middle grid
                            F_i_plus_half_x = F_i_plus_half(Fcurrentpx, F_plus_i_x, phi_plus_i_plus_half_x, F_plus_i_minus_1_x, F_minus_i_plus_1_x, phi_minus_i_plus_half_x, F_minus_i_plus_2_x)
                            F_i_minus_half_x = F_i_plus_half(Fcurrentpx, F_plus_i_minus_1_x, phi_plus_i_minus_half_x, F_plus_i_minus_2_x, F_minus_i_x, phi_minus_i_minus_half_x, F_minus_i_plus_1_x)
                            #################################################
                            # flux
                            F_plus_i_y = F_plus_minus(1, Fcurrentpy, fcurrent)
                            F_plus_i_plus_1_y = F_plus_minus(1, Fcurrentpy, frightpy)
                            F_plus_i_minus_1_y = F_plus_minus(1, Fcurrentpy, fleftpy)
                            F_plus_i_minus_2_y = F_plus_minus(1, Fcurrentpy, fleftleftpy)
                            F_minus_i_y = F_plus_minus(-1, Fcurrentpy, fcurrent)  
                            F_minus_i_minus_1_y = F_plus_minus(-1, Fcurrentpy, fleftpy)
                            F_minus_i_plus_1_y = F_plus_minus(-1, Fcurrentpy, frightpy)
                            F_minus_i_plus_2_y = F_plus_minus(-1, Fcurrentpy, frightrightpy)

                            # flux limiter
                            phi_plus_i_plus_half_y = phi_plus(F_plus_i_minus_1_y, F_plus_i_y, F_plus_i_plus_1_y, theta)
                            phi_minus_i_plus_half_y = phi_minus(F_minus_i_y, F_minus_i_plus_1_y, F_minus_i_plus_2_y, theta)
                            phi_plus_i_minus_half_y = phi_plus(F_plus_i_minus_2_y, F_plus_i_minus_1_y, F_plus_i_y, theta)
                            phi_minus_i_minus_half_y = phi_minus(F_minus_i_minus_1_y, F_minus_i_y, F_minus_i_plus_1_y, theta)

                            # flux at the middle grid
                            F_i_plus_half_y = F_i_plus_half(Fcurrentpy, F_plus_i_y, phi_plus_i_plus_half_y, F_plus_i_minus_1_y, F_minus_i_plus_1_y, phi_minus_i_plus_half_y, F_minus_i_plus_2_y)
                            F_i_minus_half_y = F_i_plus_half(Fcurrentpy, F_plus_i_minus_1_y, phi_plus_i_minus_half_y, F_plus_i_minus_2_y, F_minus_i_y, phi_minus_i_minus_half_y, F_minus_i_plus_1_y)

                            ####################################################
                            # flux
                            F_plus_i_z = F_plus_minus(1, Fcurrentpz, fcurrent)
                            F_plus_i_plus_1_z = F_plus_minus(1, Fcurrentpz, frightpz)
                            F_plus_i_minus_1_z = F_plus_minus(1, Fcurrentpz, fleftpz)
                            F_plus_i_minus_2_z = F_plus_minus(1, Fcurrentpz, fleftleftpz)
                            F_minus_i_z = F_plus_minus(-1, Fcurrentpz, fcurrent)  
                            F_minus_i_minus_1_z = F_plus_minus(-1, Fcurrentpz, fleftpz)
                            F_minus_i_plus_1_z = F_plus_minus(-1, Fcurrentpz, frightpz)
                            F_minus_i_plus_2_z = F_plus_minus(-1, Fcurrentpz, frightrightpz)

                            # flux limiter
                            phi_plus_i_plus_half_z = phi_plus(F_plus_i_minus_1_z, F_plus_i_z, F_plus_i_plus_1_z, theta)
                            phi_minus_i_plus_half_z = phi_minus(F_minus_i_z, F_minus_i_plus_1_z, F_minus_i_plus_2_z, theta)
                            phi_plus_i_minus_half_z = phi_plus(F_plus_i_minus_2_z, F_plus_i_minus_1_z, F_plus_i_z, theta)
                            phi_minus_i_minus_half_z = phi_minus(F_minus_i_minus_1_z, F_minus_i_z, F_minus_i_plus_1_z, theta)

                            # flux at the middle grid
                            F_i_plus_half_z = F_i_plus_half(Fcurrentpz, F_plus_i_z, phi_plus_i_plus_half_z, F_plus_i_minus_1_z, F_minus_i_plus_1_z, phi_minus_i_plus_half_z, F_minus_i_plus_2_z)
                            F_i_minus_half_z = F_i_plus_half(Fcurrentpz, F_plus_i_minus_1_z, phi_plus_i_minus_half_z, F_plus_i_minus_2_z, F_minus_i_z, phi_minus_i_minus_half_z, F_minus_i_plus_1_z)

                            V_sum_second = dt/dpx[p_type]*(F_i_plus_half_x-F_i_minus_half_x) + dt/dpy[p_type]*(F_i_plus_half_y-F_i_minus_half_y) + dt/dpz[p_type]*(F_i_plus_half_z-F_i_minus_half_z)
                            
                            ####################################################
                            ####################################################
                            f_drift = fcurrent + dt*collision_term[i_level, p_type, i_grid] - V_sum_second
                            
                        elif whether_drift < 0.5 and whether_Vlasov < 0.5:
                            f_drift = fcurrent + dt*collision_term[i_level, p_type, i_grid]
                            
        
                        f_updated[i_level, p_type, i_grid] = f_drift

            
# We use absorption boundary only, this is decrapted
#   
#############################################################################################################

#                     # corrections due to the physical boundary, we consider reflection and absorption or in between
#                     if vx > 0:
#                         # if vx > 0, the right most boundary needs to be considered
#                         if ix > (nx - 1.5):
#                             # mirror index for ipx
#                             ipx_mirror = mirror(npx, ipx)
#                             i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx_mirror, ipy, ipz, nx, ny, nz, npx, npy, npz)

#                             cuda.atomic.add(f_updated, (i_level, p_type, i_grid_mirror), UPFD(dx, vx, fcurrent, 0)/(denominator)*(1-x_bound_config[iy, iz]))

#                     else:
#                         # if vx < 0., the left most boundary needs to be considered                    
#                         if ix < 0.5:
#                             # mirror in dex for ipx
#                             ipx_mirror = mirror(npx, ipx)
#                             i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx_mirror, ipy, ipz, nx, ny, nz, npx, npy, npz)

#                             cuda.atomic.add(f_updated, (i_level, p_type, i_grid_mirror), UPFD(dx, vx, 0, fcurrent)/(denominator)*(1-x_bound_config[iy, iz]))

#     ##########################################################################################################
#                     if vy > 0:                    
#                         # if vy > 0, the right most boundary needs to be considered
#                         if iy > (ny - 1.5):
#                             # mirror index for ipy
#                             ipy_mirror = mirror(npy, ipy)
#                             i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy_mirror, ipz, nx, ny, nz, npx, npy, npz)

#                             cuda.atomic.add(f_updated, (i_level, p_type, i_grid_mirror), UPFD(dy, vy, fcurrent, 0)/(denominator)*(1-y_bound_config[iz, ix]))

#                     else:
#                         # if vy < 0., the left most boundary needs to be considered                   
#                         if iy < 0.5:
#                             # mirror index for ipy
#                             ipy_mirror = mirror(npy, ipy)
#                             i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy_mirror, ipz, nx, ny, nz, npx, npy, npz)

#                             cuda.atomic.add(f_updated, (i_level, p_type, i_grid_mirror), UPFD(dy, vy, 0, fcurrent)/(denominator)*(1-y_bound_config[iz, ix]))

#     ###########################################################################################################
#                     if vz > 0:
#                         # if vz > 0, the right most boundary needs to be considered
#                         if iz > (nz - 1.5):
#                             # mirror index for ipz
#                             ipz_mirror = mirror(npz, ipz)
#                             i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipz_mirror, nx, ny, nz, npx, npy, npz)

#                             cuda.atomic.add(f_updated, (i_level, p_type, i_grid_mirror), UPFD(dz, vz, fcurrent, 0)/(denominator)*(1-z_bound_config[ix, iy]))

#                     else:
#                         # if vz < 0., the left most boundary needs to be considered
#                         if iz < 0.5:
#                             # mirror index for ipz
#                             ipz_mirror = mirror(npz, ipz)
#                             i_grid_mirror = threeD_to_oneD(ix, iy, iz, ipx, ipy, ipz_mirror, nx, ny, nz, npx, npy, npz)

#                             cuda.atomic.add(f_updated, (i_level, p_type, i_grid_mirror), UPFD(dz, vz, 0, fcurrent)/(denominator)*(1-z_bound_config[ix, iy]))

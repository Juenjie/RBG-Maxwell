import math, numba
import numpy as np

def obtain_particle_numbers(f, dx, dy, dz, dpx, dpy, dpz):
    '''
    Return the particle numbers of the current distributions'''

    # the volume element associated the integration
    phase_space_volume_element = dx*dy*dz*dpz*dpy*dpz/(2*math.pi)**3

    return np.array([np.sum(f[i]) for i in range(7)])*phase_space_volume_element

def obtain_particle_charges(particle_number, charges):
    '''
    Return the particle numbers of the current distributions'''

    return [particle_number[i]*charges[i] for i in range(len(particle_number))]

def obtain_particle_energy(f, dx, dy, dz, dpx, dpy, dpz,
                           px_grid_size,
                           py_grid_size,
                           pz_grid_size,
                           px_left_bound, 
                           py_left_bound, 
                           pz_left_bound,
                           masses):
    '''
    Return the particle energys of the current distributions'''

    # the volume element associated the integration
    phase_space_volume_element = dx*dy*dz*dpz*dpy*dpz/(2*math.pi)**3
    
    f = f.sum(axis=(1,2,3))
    length = len(masses)
    particle_energies = np.zeros(length)
    
    for ipx in range(px_grid_size):
        px = (ipx+0.5)*dpx + px_left_bound
        for ipy in range(py_grid_size):
            py = (ipy+0.5)*dpy + py_left_bound
            for ipz in range(pz_grid_size):
                pz = (ipz+0.5)*dpz + pz_left_bound
                
                E = [math.sqrt(masses[i]**2+px**2+py**2+pz**2) for i in range(length)]
                
                for i_type in range(length):
                    particle_energies[i_type] += f[i_type,ipx,ipy,ipz]*E[i_type]
    
    return np.array([particle_energies[i] for i in range(len(masses))])*phase_space_volume_element

@numba.jit
def acquire_distribution_of_E(f_x_p_t, mass_squared,
                              px_grid_size,
                              py_grid_size,
                              pz_grid_size,
                              px_left_bound, py_left_bound, pz_left_bound,
                              dpx, dpy, dpz,
                              p_type,ix, iy, iz,
                              round_bin_digit):
    
    '''
    Give the distribution in terms of energy E.

    Params
    ======
    f_x_p_t:
        distribution of shape 
        [7,x_grid_size, y_grid_size, z_grid_size,
            half_px_grid_size*2,half_py_grid_size*2,half_pz_grid_size*2]
    mass_squared: 
        the mass squared of the current species at current spatial grid
    px_grid_size, py_grid_size, pz_grid_size: 
        number of grid sizes in in momentum domain, e.g., [5,5,5]
    px_left_bound, py_left_bound, pz_left_bound:
        left boundary of the momentum region, in unit GeV
    dpx, dpy, dpz:
        infinitesimal difference in momentum coordinate, these are in GeV (recomended)
    p_type:
        integer between 0-6, specify which species to calculate
    ix, iy, iz: 
        integers, specify which spatial grid to calculate
    round_bin_digit:
        integer, the round used for round the energys. The larger the integer,
        the smaller histogram bin will be.

    Return
    ======
    E,f: 
        lists, the energy with corresponding distribution
    '''
    
    # greate a dictionary with energy as keys
    # nE keeps track of the number of points, since we will take the average of f(E)
    f_collect = []
    E_collect = []
    
    # loop through momentum grids
    for ipx in range(px_grid_size):
        for ipy in range(py_grid_size):
            for ipz in range(pz_grid_size):
                
                # central value for each grid 
                px = (ipx+0.5)*dpx + px_left_bound
                py = (ipy+0.5)*dpy + py_left_bound
                pz = (ipz+0.5)*dpz + pz_left_bound
                E = np.round(math.sqrt(mass_squared+px**2+py**2+pz**2),round_bin_digit)
                
                # accumulate the distribution values
                E_collect.append(E)
                f_collect.append(f_x_p_t[p_type,ix,iy,iz,ipx,ipy,ipz])
        
    return E_collect, f_collect


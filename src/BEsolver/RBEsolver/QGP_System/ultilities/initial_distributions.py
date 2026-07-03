#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
def CGC_states(dpx, dpy, dpz, 
               px_grid_size = 10,
               py_grid_size = 10,
               pz_grid_size = 10,
               x_grid_size = 1,
               y_grid_size = 1,
               z_grid_size = 11,
               px_left_bound = -2, 
               py_left_bound = -2, 
               pz_left_bound = -2,
               Qs = 1,
               xi = 1.4,
               f0g = 0.5,
               f0u = 0,
               f0d = 0,
               f0s = 0,
               f0ubar = 0,
               f0dbar = 0,
               f0sbar = 0):
    
    '''
    The function gives an initial function following the paper  
    Chemical equilibration in weakly coupled QCD.
    
    fg(px,py,pz) = f0*\theta(1-math.sqrt(px**2+py**2+pz**2*xi**2)/Qs)
    
    Params
    ======
    dpx, dpy, dpz:
        infinitesimal difference in momentum coordinate, these are in GeV (recomended)
    x_grid_size, y_grid_size, z_grid_size:
        number of grids in spatial domain, e.g., [5, 5, 5]
    px_grid_size, py_grid_size, pz_grid_size: 
        number of grid sizes in in momentum domain, e.g., [5,5,5]
    px_left_bound, py_left_bound, pz_left_bound:
        left boundary of the momentum region, in unit GeV    
    x_bound_id:
        index for initial spatial grids where the particles are filled.
    Qs: 
        saturation scale
    xi:
        unsymmetricality
    f0g:
        gluon occupation
    f0u, f0d, f0s, f0ubar, f0dbar, f0sbar:
        quark occupation
    
    Return
    ======
    initial_distribution: 
        shape: [7, x_grid_size, y_grid_size, z_grid_size,
                   px_grid_size,py_grid_size,pz_grid_size]        
    '''
    
    shape_of_distribution_data = [7, x_grid_size, y_grid_size, z_grid_size,
                                     px_grid_size,py_grid_size,pz_grid_size]
      
    initial_distribution = np.zeros(shape_of_distribution_data)
    
    for ipx in range(px_grid_size):
        px = (ipx+0.5)*dpx + px_left_bound
        for ipy in range(py_grid_size):
            py = (ipy+0.5)*dpy + py_left_bound
            for ipz in range(pz_grid_size):
                pz = (ipz+0.5)*dpz + pz_left_bound
                
                para = 1 - math.sqrt(px**2+py**2+pz**2*xi**2)/Qs
                if para >= 0.:
                    # only put the particles at the center
                    initial_distribution[6, int(x_grid_size/2), int(y_grid_size/2), int(z_grid_size/2),ipx, ipy, ipz] = f0g
                    initial_distribution[0, int(x_grid_size/2), int(y_grid_size/2), int(z_grid_size/2),ipx, ipy, ipz] = f0u
                    initial_distribution[1, int(x_grid_size/2), int(y_grid_size/2), int(z_grid_size/2),ipx, ipy, ipz] = f0d
                    initial_distribution[2, int(x_grid_size/2), int(y_grid_size/2), int(z_grid_size/2),ipx, ipy, ipz] = f0s
                    initial_distribution[3, int(x_grid_size/2), int(y_grid_size/2), int(z_grid_size/2),ipx, ipy, ipz] = f0ubar
                    initial_distribution[4, int(x_grid_size/2), int(y_grid_size/2), int(z_grid_size/2),ipx, ipy, ipz] = f0dbar
                    initial_distribution[5, int(x_grid_size/2), int(y_grid_size/2), int(z_grid_size/2),ipx, ipy, ipz] = f0sbar

    return initial_distribution

def OAM_CGC_Au_Au_states(dpx, dpy, dpz, 
                         dx, x_left_bound,
                         dz, z_left_bound,
                         px_grid_size = 10,
                         py_grid_size = 10,
                         pz_grid_size = 10,
                         x_grid_size = 11,
                         y_grid_size = 1,
                         z_grid_size = 11,
                         px_left_bound = -2, 
                         py_left_bound = -2, 
                         pz_left_bound = -2,
                         Qs = 1,
                         xi = 1.4,
                         f0g = 7.880,
                         f0u = 0.630,
                         f0d = 0.721,
                         f0s = 0,
                         f0ubar = 0,
                         f0dbar = 0,
                         f0sbar = 0,
                         masses = None, # GeV
                         s = 200**2, # GeV^2
                         b = None,
                         rhoAu = None,
                         tilted_angle = math.pi/180*10): # GeV^-1
    
    '''
    The function gives an initial function following the paper  
    Chemical equilibration in weakly coupled QCD. The function is limited to XOZ plane.
    The function gives the initial distribution before t=0 of the participants. That means two half plates.
    
    fg(px,py,pz) = f0*\theta(1-math.sqrt(px**2+py**2+pz**2*xi**2)/Qs)
    
    Params
    ======
    dpx, dpy, dpz:
        infinitesimal difference in momentum coordinate, these are in GeV (recomended)
    x_grid_size, y_grid_size, z_grid_size:
        number of grids in spatial domain, e.g., [5, 5, 5]
    px_grid_size, py_grid_size, pz_grid_size: 
        number of grid sizes in in momentum domain, e.g., [5,5,5]
    px_left_bound, py_left_bound, pz_left_bound:
        left boundary of the momentum region, in unit GeV    
    x_bound_id:
        index for initial spatial grids where the particles are filled.
    Qs: 
        saturation scale
    xi:
        unsymmetricality
    f0g:
        gluon occupation
    f0u, f0d, f0s, f0ubar, f0dbar, f0sbar:
        quark occupation
    masses = [0.5,0.5,0.5,0.5,0.5,0.5,0.7] GeV
    s = 200**2 GeV^2
    b = 4/0.197 GeV^-1
    
    Return
    ======
    initial_distribution: 
        shape: [7, x_grid_size, y_grid_size, z_grid_size,
                   px_grid_size,py_grid_size,pz_grid_size]        
    '''
    
    exec(rhoAu, globals())
    
    y_grid_size = 1
    shape_of_distribution_data = [7, x_grid_size, y_grid_size, z_grid_size,
                                     px_grid_size,py_grid_size,pz_grid_size]
      
    initial_distribution = np.zeros(shape_of_distribution_data)
    
    # this only holds for Au
    R0 = 6.42/0.197 #(*GeV^-1*)
    RA = 7.019/0.197 #(*GeV^-1*)
    mN = 0.938 #(*GeV, nucleon mass*)
    v = math.sqrt(1-(2*mN/math.sqrt(s))**2)
    
    gamma = np.zeros([len(masses), x_grid_size])
    beta_z = np.zeros([len(masses), x_grid_size])
    n_x = np.zeros([len(masses), x_grid_size])
    
    # first calculate gamma and beta_z in each x-grid
    for ix in range(x_grid_size):
        x = dx*(ix + 0.5) + x_left_bound
        
        for i_type in range(len(masses)):
            # if particle numbers at current point are positive
            n_temp = dx*(rho(x-b/2)+rho(x+b/2))
            if n_temp >0:
                n_x[i_type, ix] = n_temp
                # net momentum at the current point for each particle in everage
                p_z_x = dx*((rho(x-b/2)-rho(x+b/2))*masses[i_type]*v)/n_temp
                # net momentum in a spatial grid x-aixs
                beta_z[i_type, ix] = p_z_x/math.sqrt(p_z_x**2+masses[i_type]**2)
                gamma[i_type, ix] = 1/math.sqrt(1-beta_z[i_type, ix]**2)

    n_ratio = [n_x[i]/np.sum(n_x, 1)[i] for i in range(len(masses))]
    n_ratio = [n_ratio[i]/max(n_ratio[i]) for i in range(len(masses))]
    
    x_region = RA - b/2
    # we only support overlap collisions
    if x_region<=0:
        raise AssertionError("no overlapped regions are found!")

    # give the momentum distribution at each spatial grid according to the Lorentz boost
    for ipx in range(px_grid_size):
        px = (ipx+0.5)*dpx + px_left_bound
        for ipy in range(py_grid_size):
            py = (ipy+0.5)*dpy + py_left_bound
            for ipz in range(pz_grid_size):
                pz = (ipz+0.5)*dpz + pz_left_bound
                
                for ix in range(x_grid_size):
                    x = (ix+0.5)*dx + x_left_bound
                    
                    # only feed data in the overlapped region
                    if x < x_region and x > -x_region:
                        
                        # rotate axis
                        zp = -x*math.sin(tilted_angle)
                        xp = x*math.cos(tilted_angle)
                        pzp = pz*math.cos(tilted_angle)-px*math.sin(tilted_angle)
                        pxp = pz*math.sin(tilted_angle)+px*math.cos(tilted_angle)
                        
                        if abs(pzp) < abs(pz_left_bound) and abs(pxp) < abs(px_left_bound) and\
                           abs(zp) < abs(z_left_bound) and abs(xp) < abs(x_left_bound):
                            
                            # find the index for zp, xp, pzp, pxp
                            ixp = int((xp +abs(x_left_bound))/dx)
                            izp = int((zp +abs(z_left_bound))/dz)
                            ipxp = int((pxp +abs(px_left_bound))/dpx)
                            ipzp = int((pzp +abs(pz_left_bound))/dpz)
                            
                            pz_prime = -gamma[-1, ix]*beta_z[-1, ix]*math.sqrt(px**2+py**2+pz**2+masses[-1]**2)\
                                       +gamma[-1, ix]*pz
                            para = 1 - math.sqrt(px**2+py**2+pz_prime**2*xi**2)/Qs
                            if para >= 0.: 
                                initial_distribution[-1, ixp, 0, izp,\
                                                        ipxp, ipy, ipzp] = f0g*n_ratio[-1][ix]

                            pz_prime = -gamma[0, ix]*beta_z[0, ix]*math.sqrt(px**2+py**2+pz**2+masses[0]**2)\
                                       +gamma[0, ix]*pz
                            para = 1 - math.sqrt(px**2+py**2+pz_prime**2*xi**2)/Qs
                            if para >= 0.: 
                                initial_distribution[0, ixp, 0, izp,\
                                                        ipxp, ipy, ipzp] = f0u*n_ratio[0][ix]

                            pz_prime = -gamma[1, ix]*beta_z[1, ix]*math.sqrt(px**2+py**2+pz**2+masses[1]**2)\
                                       +gamma[1, ix]*pz
                            para = 1 - math.sqrt(px**2+py**2+pz_prime**2*xi**2)/Qs
                            if para >= 0.:
                                initial_distribution[1, ixp, 0, izp,\
                                                        ipxp, ipy, ipzp] = f0d*n_ratio[1][ix]
    return initial_distribution

import numpy as np
import math
def gaussian_states(half_px_grid_size = 20,
                    half_py_grid_size = 20,
                    half_pz_grid_size = 20,
                    x_grid_size = 1,
                    y_grid_size = 1,
                    z_grid_size = 1,
                    half_x = 3,
                    half_y = 3,
                    half_z = 3,
                    half_px = 2,
                    half_py = 2,
                    half_pz = 2,
                    mu = 1.0,
                    sigma = 0.2,
                    amplitude = 0.5,
                    p_type = 6):
    
    '''
    The function gives an initial function of Gaussian type.
    
    fg(p) = amplitude*exp(-(p-mu)**2/(2*sigma**2))
    
    mu is taken to be 0.2 GeV,
    sigma is 0.2 GeV
    and amplitude is taken 0.5 as default
    
    >>>
    argument:
        half_px_grid_size, half_py_grid_size, half_pz_grid_size:
            momentum 3-d box grid size. This the the half value.
        x_grid_size, y_grid_size, z_grid_size:
            spatial 3-d box grid size
        half_px, half_py, half_pz:
            the half momentum box length. the real range of the momentum box is
            [[-half_px,half_px],[-half_py,half_py],[-half_pz,half_pz]]
        half_x, half_y, half_z:
            the half spatial box length. the real range of the momentum box is
            [[-half_x,half_x],[-half_y,half_y],[-half_z,half_z]]
        mu:
            the central value of the momentum, 0.2 GeV as default
        sigma:
            the width of the distribution, 0.2 GeV as default
        amplitude:
            maximum value of f, 0.5 as default
        p_type:
            int, the species that needs to be generated
    >>>
    return:
        initial_distribution: 
            numpy array, shape:
            [7, x_grid_size, y_grid_size, z_grid_size,
                half_px_grid_size*2,half_py_grid_size*2,half_pz_grid_size*2]        
    '''
    # force spatial grid size to be 1
    if x_grid_size != 1 or y_grid_size != 1 or z_grid_size != 1:
        raise AssertionError("make sure that x_grid_size = 1 and y_grid_size == 1 and z_grid_size == 1")
    
    shape_of_distribution_data = [7, x_grid_size, y_grid_size, z_grid_size,
                                     half_px_grid_size*2,half_py_grid_size*2,half_pz_grid_size*2]
      
    initial_distribution = np.zeros(shape_of_distribution_data)
    
    # momentum grid size
    px_grid_size, py_grid_size, pz_grid_size = 2*half_px_grid_size, 2*half_py_grid_size, 2*half_pz_grid_size
    
    dpx, dpy, dpz = half_px/half_px_grid_size, half_py/half_py_grid_size, half_pz/half_pz_grid_size # GeV
    
    for ipx in range(px_grid_size):
        for ipy in range(py_grid_size):
            for ipz in range(pz_grid_size):
                
                # central value for each grid 
                px = (ipx+0.5)*dpx - half_px
                py = (ipy+0.5)*dpy - half_py
                pz = (ipz+0.5)*dpz - half_pz
                
                p = math.sqrt(px**2+py**2+pz**2)
                initial_distribution[p_type,0,0,0,ipx,ipy,ipz] =                             amplitude*math.exp(-(p-mu)**2/(2*sigma**2))

    return initial_distribution

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
def left_low_right_high_distribution(half_px_grid_size = 1,
                                     half_py_grid_size = 5,
                                     half_pz_grid_size = 5,
                                     x_grid_size = 10,
                                     y_grid_size = 10,
                                     z_grid_size = 10,
                                     half_x = 3,
                                     half_px = 2,
                                     half_y = 3,
                                     half_py = 2,
                                     half_z = 3,
                                     half_pz = 2,
                                     f_min = 0.5,
                                     f_max = 0.99,
                                     Qs = 1.,
                                     f0g = 0.2,
                                     f0q = 0.2,
                                     dpx = 0.3, dpy = 0.3, dpz = 0.3):
    
    '''
    Give the distribution for gluons only.
    This function generate a distribution function which has less particles in the left (x=-half_x), 
    than in the right (x=half_x), where 2*half_x is the length of spatial box in x-direction.
    >>>
    argument:
        half_px_grid_size, half_py_grid_size, half_pz_grid_size:
            momentum 3-d box grid size. This the the half value.
            
        x_grid_size, y_grid_size, z_grid_size:
            spatial 3-d box grid size
            
        half_px, half_py, half_pz:
            the half momentum box length. the real range of the momentum box is
            [[-half_px,half_px],[-half_py,half_py],[-half_pz,half_pz]]
            
        half_x, half_y, half_z:
            the half spatial box length. the real range of the momentum box is
            [[-half_x,half_x],[-half_y,half_y],[-half_z,half_z]]
            
        f_min: 
            the approximate minimum distribution value for all grids.
            
        f_max: 
            the approximate maximum distribution value for all grids.
    >>>
    return:
        initial_distribution: 
            numpy array, shape:
            [7, x_grid_size, y_grid_size, z_grid_size,
                half_px_grid_size*2,half_py_grid_size*2,half_pz_grid_size*2]        
    '''
        
    dz = half_z/0.197*2/z_grid_size # GeV^-1
    dpz = half_pz/half_pz_grid_size # GeV
    
    # momentum grid size
    px_grid_size, py_grid_size, pz_grid_size = 2*half_px_grid_size, 2*half_py_grid_size, 2*half_pz_grid_size
    
    # position increase and momentum decrease
    slope_z = (f_max-f_min)/(2*half_z/0.197) # GeV
    
    shape_of_distribution_data = [7, x_grid_size, y_grid_size, z_grid_size,
                                     half_px_grid_size*2,half_py_grid_size*2,half_pz_grid_size*2]
    initial_distribution = np.zeros(shape_of_distribution_data)
    
    for ipx in range(px_grid_size):
        for ipy in range(py_grid_size):
            for ipz in range(pz_grid_size):
                
                # central value for each grid 
                px = (ipx+0.5)*dpx - half_px
                py = (ipy+0.5)*dpy - half_py
                pz = (ipz+0.5)*dpz - half_pz
                p = math.sqrt(px**2+py**2+pz**2)
                
                if p<Qs:
                    # feed in values according to spatial grid z
                    for iz in range(z_grid_size):

                            Dz = (iz + 0.5)*dz
                            
                            # feed value
                            # loop through particle species
                            for p_type in range(7):
                                if p_type == 6:
                                    initial_distribution[p_type,:,:,iz:(iz+1),ipx,ipy,ipz] = f0g*(slope_z*Dz)*np.random.random([x_grid_size,y_grid_size, 1])+2*f_min
                                else:
                                    initial_distribution[p_type,:,:,iz:(iz+1),ipx,ipy,ipz] = f0q*(slope_z*Dz)*np.random.random([x_grid_size,y_grid_size, 1])
    return initial_distribution

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
def step_function_distribution(px_grid_size = 20,
                               py_grid_size = 20,
                               pz_grid_size = 20,
                               x_grid_size = 1,
                               y_grid_size = 1,
                               z_grid_size = 1,
                               half_x = 3,
                               half_y = 3,
                               half_z = 3,
                               half_px = 2,
                               half_py = 2,
                               half_pz = 2,
                               f0g = 0.2,
                               f0q = 0.2,
                               Qs = 1.,
                               species = [0,0,0,0,0,0,1]):
    
    '''
    The function gives a step distribution function with the formula
    f(0,p) = f0*\theta(1-p/Qs), usually Qs ~ 1 GeV,
    for |p|<Qs, f(0,p) = f0, otherwise f(0,p) = 0.
    Only specified species will be given the value.
    >>>
    argument:
        half_px_grid_size, half_py_grid_size, half_pz_grid_size:
            momentum 3-d box grid size. This the the half value.
        x_grid_size, y_grid_size, z_grid_size:
            spatial 3-d box grid size
        half_px, half_py, half_pz:
            the half momentum box length. the real range of the momentum box is
            [[-half_px,half_px],[-half_py,half_py],[-half_pz,half_pz]]
        half_x, half_y, half_z:
            the half spatial box length. the real range of the momentum box is
            [[-half_x,half_x],[-half_y,half_y],[-half_z,half_z]]
        f0g: 
            the amplitude for gluon distribution grid values.
        f0q: 
            the amplitude for fermion distribution grid values.
        Qs: 
            saturation scale, typical value is around 1 GeV
        species:
            list with 7 elements of integer number 0/1.
            stands for u,d,s,ubar,dbar,sbar,g.
            1 means that this species is given the initial value,
            and 0 means this species is not given the value.
    >>>
    return:
        initial_distribution: 
            numpy array, shape:
            [7, x_grid_size, y_grid_size, z_grid_size,
                half_px_grid_size*2,half_py_grid_size*2,half_pz_grid_size*2]        
    '''
    
    # force spatial grid size to be 1
    if x_grid_size != 1 or y_grid_size != 1 or z_grid_size != 1:
        raise AssertionError("make sure that x_grid_size = 1 and y_grid_size == 1 and z_grid_size == 1")
    
    shape_of_distribution_data = [7, x_grid_size, y_grid_size, z_grid_size,
                                     px_grid_size,py_grid_size,pz_grid_size]
      
    initial_distribution = np.zeros(shape_of_distribution_data)
    
    # momentum grid size
    dpx, dpy, dpz = 2*half_px/px_grid_size, 2*half_py/py_grid_size, 2*half_pz/pz_grid_size # GeV
    
    for ipx in range(px_grid_size):
        for ipy in range(py_grid_size):
            for ipz in range(pz_grid_size):
                
                # central value for each grid 
                px = (ipx+0.5)*dpx - half_px
                py = (ipy+0.5)*dpy - half_py
                pz = (ipz+0.5)*dpz - half_pz
                p = math.sqrt(px**2+py**2+pz**2)
                if p<Qs:
                    
                    # loop through particle species
                    for p_type in range(7):
                        if p_type == 6:
                            initial_distribution[p_type,0,0,0,ipx,ipy,ipz] = f0g*species[p_type]
                        else:
                            initial_distribution[p_type,0,0,0,ipx,ipy,ipz] = f0q*species[p_type]
                
    return initial_distribution


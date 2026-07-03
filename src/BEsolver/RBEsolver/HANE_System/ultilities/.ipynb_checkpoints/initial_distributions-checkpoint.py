#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math

def center_expand_states(dpx, dpy, dpz, 
                         dx, x_left_bound,
                         dy, y_left_bound,
                         dz, z_left_bound,
                         px_grid_size = 11,
                         py_grid_size = 11,
                         pz_grid_size = 11,
                         x_grid_size = 11,
                         y_grid_size = 11,
                         z_grid_size = 11,
                         px_left_bound = -2, 
                         py_left_bound = -2, 
                         pz_left_bound = -2,
                         fFe = 0.5,
                         fO = 0,
                         fe = 0,
                         # https://www.americanelements.com/
                         # Fe-56 55.935 and O-16 15.9994
                         masses = [(55.935*931-0.511)*10**6,(15.9994*931-0.511)*10**6,0.511*10**6]):# eV, Fe, O, e

    """
    Initial conditions for electrons, oxygen and iron.
    Electrons and oxygen are every the same with Maxwellian distributions of 
    Temperature 300 Kelvin which is 2.5852*10**-8 MeV, this gives a velocity of scale 10**-6
    Irons are at the center spatial grid with velocity 1/150 c ~ 2*10**6 m/s.
    Since px_left_bound ~ - 350 MeV, the momentum resolution cannot distinguish the velocity for O and e.
    """
    shape_of_distribution_data = [3, x_grid_size, y_grid_size, z_grid_size,
                                     px_grid_size,py_grid_size,pz_grid_size]
      
    initial_distribution = np.zeros(shape_of_distribution_data)
    
    # electrons and oxygen have zero momentum and all spatial distributions
    initial_distribution[1,:,:,:, int(px_grid_size/2),int(py_grid_size/2),int(pz_grid_size/2)] = fO
    initial_distribution[2,:,:,:, int(px_grid_size/2),int(py_grid_size/2),int(pz_grid_size/2)] = fe
    
    # the particles are in the momentum range
    pRange = [250*10**6, 450*10**6] #eV
    
    # give the momentum distribution at central spatial grid 
    for ipx in range(px_grid_size):
        px = (ipx+0.5)*dpx + px_left_bound
        for ipy in range(py_grid_size):
            py = (ipy+0.5)*dpy + py_left_bound
            for ipz in range(pz_grid_size):
                pz = (ipz+0.5)*dpz + pz_left_bound
                
                p = math.sqrt(px**2+py**2+pz**2+masses[0]**2)
                
                if p > pRange[0] and p < pRange[1]:
                    initial_distribution[0,int(x_grid_size/2),int(y_grid_size/2),int(z_grid_size/2),ipx,ipy,ipz] = \
                    fFe
    
    return initial_distribution

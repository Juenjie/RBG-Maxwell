import cupy
import math

def check_legacy_of_distributions(distribution, particle_type, hbar, dt, degeneracy):
    '''
    Check if the particle distributions are always larger than 0.
    For fermions, the distribution function should not larger than 1.
    
    Params
    ======
    distribution: distributions of the particles, of shape [momentum_levels, particle_species, nx*ny*nz*npx*npy*npz] 
    
    Return
    ======
    True for legal and False for illegal
    '''
    length = len(degeneracy)
    
    # loop thorugh the particle types
    for i_type in range(length):
        
        # take the minimum value of the distribution
        f_min = cupy.amin(distribution[:,i_type])
        
        # if the distribution is smaller than zero, return False
        # very small statistical negative distributiona are legal
        if f_min < -10**(-19):
            return False,'f_min',f_min,i_type

#         # for fermions we need to find the maximum value of the distributions as well
#         if particle_type[i_type] == 1:
#             f_max = cupy.amax(distribution[:,i_type])
#             # return False if the maximum value is larger than g*1/(2*math.pi*hbar)**3 for fermions
#             if f_max > degeneracy[i_type]*1/(2*math.pi*hbar)**3:
#                 return False,'f_max',f_max,i_type
        
    # if no issues are found, return True
    return True,0,0,0
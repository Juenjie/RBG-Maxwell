import cupy

def check_legacy_of_distributions(distribution, particle_type):
    '''
    Check if the particle distributions are always larger than 0.
    For fermions, the distribution function should not larger than 1.
    
    Params
    ======
    distribution: distributions of the particles, of shape [n_type, nx, ny, nz, npx, npy, npz] 
    particle_type: should be consistent with initial_distribution, indicate whether the particle is 
                   classical (0), fermi (1) or bosonic type (2), e.g., for five particle species, [0,0,1,1,2]
                   means these five particles are [classical, classical, fermi, fermi, bosonic]
    
    Return
    ======
    True for legal and False for illegal
    '''
    
    # loop thorugh the particle types
    for i_type in range(len(particle_type)):
        
        # take the minimum value of the distribution
        f_min = cupy.amin(distribution[i_type])
        
        # if the distribution is smaller than zero, return False
        if f_min < -10**(-15):
            return False,'f_min',f_min,i_type
        
        # for fermions we need to find the maximum value of the distributions as well
        if particle_type[i_type] == 1:
            f_max = cupy.amax(distribution[i_type])
            # return False if the maximum value is larger than 1 for fermions
            if f_max > 1.:
                return False,'f_max',f_max,i_type
        
    # if no issues are found, return True
    return True,'ok','ok','ok'
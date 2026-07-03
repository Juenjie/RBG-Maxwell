import numpy as np
def collision_type_for_all_species():
    
    '''
    collision type, this is a collection of all possible combinations of scatterings involving
    Fe3+ (0), He (1).
    
    for 2-2 collisions:
        (type 0): Fe3+ + He <-> He  + Fe3+
    
    >>>
    return:
    NOTE: these outputs must be consistent with each other!!!!!!
        flavor and collision_type (2 to 2 process):
            numpy arrays.
            
            # the principle for writing the possible collisions is that the collisions
            # must be symmetrical, i.e., if we have a + b -> c + d, we must at the same time
            # consider c + d -> a + b. Note that a + a -> a + a is symmetrical already.
            
            # flavor: all possible collisions for the given final particle
            use the value 10001 to occupy the non-existed collision 
            
            # collision_type: an index indicate which collision type the process belongs to, 
            # use the value 10001 to occupy the non-existed collisioneg:
            large values indicate invalid collisions and will be excluded in the code
            
    '''

    particle_order = 'Fe3+ (0), He (1)'
    
    flavor, collision_type = {}, {}
    # initialize the values to None
    flavor['2TO2'], flavor['2TO3'], flavor['3TO2'] = None, None, None
    collision_type['2TO2'], collision_type['2TO3'], collision_type['3TO2'] = None, None, None
    
    # indicating final H+, H2, H, H2+, e-
    # use 10001 to occupy empty space
    collision_type['2TO2']=np.array([[0],[0]],dtype=np.int64)
    
    flavor['2TO2']=np.array([[[1,0,1,0]],[[1,0,1,0]]],dtype=np.int64)
    
    return flavor, collision_type, particle_order

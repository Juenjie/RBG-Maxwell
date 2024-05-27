import numpy as np
def collision_type_for_all_species():
    
    '''
    collision type, this is a collection of all possible combinations of scatterings involving
    H+ (0), H2 (1), H (2), H2+ (3), e- (4).
    
    for 2-2 collisions:
        (type 0): H+ + H2 <-> H  + H2+
        (type 1): H+ + H2 <-> H+ + H2
        (type 2): H2 + e- <-> e- + H2
            
    for 2-3 collisions:
        (type 0): H+ + H2 <-> H+ + H2+ + e-
        (type 1): H  + H2 <-> H+ + H2  + e-
        (type 2): H  + H2 <-> H  + H2+ + e-
        (type 3): e- + H2 <-> e- + H2+ + e-
    
    
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

    particle_order = 'H+ (0), H2 (1), H (2), H2+ (3), e- (4)'
    
    flavor, collision_type = {}, {}
    # initialize the values to None
    flavor['2TO2'], flavor['2TO3'], flavor['3TO2'] = None, None, None
    collision_type['2TO2'], collision_type['2TO3'], collision_type['3TO2'] = None, None, None
    
    # indicating final H+, H2, H, H2+, e-
    # use 10001 to occupy empty space
    collision_type['2TO2']=np.array([[1,10001],[1,2],[0,10001],[0,10001],[2,10001]],dtype=np.int64)
    
    flavor['2TO2']=np.array([[[1,0,1,0],    [10001,10001,10001,10001]],
                       [[0,1,0,1],    [4,1,4,1]],
                       [[0,1,3,2],    [10001,10001,10001,10001]],
                       [[0,1,2,3],    [10001,10001,10001,10001]],
                       [[1,4,1,4],    [10001,10001,10001,10001]]],dtype=np.int64)
    
    # indicating final H+, H2, H, H2+, e-
    # use 10001 to occupy empty space
    collision_type['2TO3']=np.array([[0,1,10001,10001],\
                               [1,10001,10001,10001],\
                               [2,10001,10001,10001],\
                               [0,2,3,10001],\
                               [0,1,2,3]],dtype=np.int64)
    
    flavor['2TO3']=np.array([[[0,1,3,4,0], [1,2,1,4,0],                     [10001,10001,10001,10001,10001], [10001,10001,10001,10001,10001]],
                       [[1,2,0,4,1], [10001,10001,10001,10001,10001], [10001,10001,10001,10001,10001], [10001,10001,10001,10001,10001]],
                       [[2,1,3,4,2], [10001,10001,10001,10001,10001], [10001,10001,10001,10001,10001], [10001,10001,10001,10001,10001]],
                       [[0,1,0,4,3], [1,2,2,4,3],                     [1,4,4,4,3],                     [10001,10001,10001,10001,10001]],
                       [[0,1,0,3,4], [1,2,0,1,4],                     [2,1,2,3,4],                     [1,4,3,4,4]]],dtype=np.int64)
        
    
    return flavor, collision_type, particle_order

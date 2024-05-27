import numpy as np
def collision_type_for_all_species():
    
    '''
    collision type, this is a collection of all possible combinations of scatterings involving
    a (0) and b (1).
    
    The processes involved here are:
    
    for 2-2 collisions:
        (type 0): a + a <-> a + a
        (type 1): b + b <-> b + b
        (type 2): a + b <-> a + b
            
    for 2-3 collisions:
        (type 0): a + a <-> b + b + b
        
    for 3-2 collisions:
        (type 0): b + b + b <-> a + a
    
    >>>
        flavor and collision_type:
            numpy arrays.
            
            # flavor: all possible collisions for the given final particle, eg: 
            #        for final d, we have
            #        aa->bbb (0)
            
            The corresponding flavor array is
            #        flavor=np.array([[[0,0,1,1,1]]],dtype=np.int64)
            
            # collision_type: an index indicate which collision type the process belongs to, eg:
            For final d quark case
            #                collision_type=np.array([[0]],dtype=np.int64)
            
    '''
    particle_order = 'a (0), b (1)'
    
    flavor, collision_type = {}, {}
    # initialize the values to None
    flavor['2TO2'], flavor['2TO3'], flavor['3TO2'] = None, None, None
    collision_type['2TO2'], collision_type['2TO3'], collision_type['3TO2'] = None, None, None
    
    flavor['2TO2']=np.array([[[0,0,0,0],[1,0,1,0]], \
                     [[1,1,1,1],[0,1,0,1]]],dtype=np.int32)
    collision_type['2TO2']=np.array([[0,2],[1,2]],dtype=np.int32)
    
    flavor['2TO3']=np.array([[[10001,10001,10001,10001,10001]],\
                     [[0,   0,   1,   1,   1]]],dtype=np.int32)
    collision_type['2TO3']=np.array([[10001], [0]],dtype=np.int32)
    
    flavor['3TO2']=np.array([[[1,   1,   1,   0,   0]],\
                     [[10001,10001,10001,10001,10001]]],dtype=np.int32)
    collision_type['3TO2']=np.array([[0],[10001]],dtype=np.int32)
        
    return flavor, collision_type, particle_order
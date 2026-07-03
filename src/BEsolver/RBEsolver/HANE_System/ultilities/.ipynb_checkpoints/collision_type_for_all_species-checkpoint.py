import numpy as np
def collision_type_for_all_species():
    
    '''
    collision type, this is a collection of all possible combinations of scatterings involving
    Fe+1 (0), O+1 (1), and e-1 (2)
    
    The processes involved here are:
    
    # FeFe->FeFe (0), OFe->OFe (1), eFe->eFe (2)
    
    # OO->OO (3), FeO->FeO (1), eO->eO (4)
    
    # ee->ee (5), Fee->Fee (2), Oe->Oe (4)
    
    params
    ======
       
    return
    ======
        flavor and collision_type:
            numpy arrays.
            
            # flavor: all possible collisions for the given final particle, eg: 
            #        for final Fe, we have
            #        FeFe->FeFe (0), OFe->OFe (1), eFe->eFe (2)
            
            The corresponding flavor array is
            #        flavor=np.array([[[0,0,0,0],[1,0,1,0],[2,0,2,0]]],dtype=np.int32)
            
            # collision_type: an index indicate which collision type the process belongs to, eg:
            For final d quark case
            #                collision_type=np.array([[0,1,2]],dtype=np.int32)
            
            where 0,1,2,3,4,5 corresponds to the following processes:
            (0): FeFe <-> FeFe
            (1): FeO <-> FeO
            (2): eFe <-> eFe
            (3): OO <-> OO
            (4): eO <-> eO
            (5): ee <-> ee
            
    '''
    
    flavor=np.array([[[0,0,0,0],[1,0,1,0],[2,0,2,0]], \
                     [[1,1,1,1],[0,1,0,1],[2,1,2,1]], \
                     [[2,2,2,2],[0,2,0,2],[1,2,1,2]]],dtype=np.int32)

    collision_type=np.array([[0,1,2], \
                             [3,1,4], \
                             [5,2,4]],dtype=np.int32)
        
    return flavor, collision_type

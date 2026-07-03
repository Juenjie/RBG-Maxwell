import numpy as np
def collision_type_for_all_species():
    
    '''
    collision type, this is a collection of all possible combinations of scatterings involving
    u,d,s,u_bar,d_bar and s_bar.
    
    The processes involved here are:
    
    # uu->uu (1), du->du (0), su->su (0), d_bar+u->d_bar+u (0), s_bar+u->s_bar+u (0), gu->gu (5)
    # u_bar+u->u_bar+u (2), d_bar+d->u_bar+u (3), s_bar+s->u_bar+u (3), gg->u_bar+u (4)
    
    # ud->ud (0), dd->dd (1), sd->sd (0), u_bar+d->u_bar+d (0), s_bar+d->s_bar+d (0), gd->gd (5)
    # d_bar+d->d_bar+d (2), u_bar+u->d_bar+d (3), s_bar+s->d_bar+d (3), gg->d_bar+d (4)
    
    # us->us (0), ds->ds (0), ss->ss (1), u_bar+s->u_bar+s (0), d_bar+s->d_bar+s (0), gs->gs (5)
    # s_bar+s->s_bar+s (2), u_bar+u->s_bar+s (3), d_bar+d->s_bar+s (3), gg->s_bar+s (4)
    
    # d+u_bar->d+u_bar (0), s+u_bar->s+u_bar (0), u_bar+u_bar->+u_bar+u_bar (1), 
    # d_bar+u_bar->d_bar+u_bar (0), s_bar+u_bar->s_bar+u_bar (0), g+u_bar->g+u_bar (5)
    # u+u_bar->u+u_bar (2), d+d_bar->u+u_bar (3), s+s_bar->u+u_bar (3), gg->u+u_bar (4)
    
    # u+d_bar->u+d_bar (0), s+d_bar->s+d_bar (0), u_bar+d_bar->u_bar+d_bar (0), 
    # d_bar+d_bar->d_bar+d_bar (1), s_bar+d_bar->s_bar+d_bar (0), g+d_bar->g+d_bar (5)
    # d+d_bar->d+d_bar (2), u+u_bar->d+d_bar (3), s+s_bar->d+d_bar (3), gg->d+d_bar (4)
    
    # u+s_bar->u+s_bar (0), d+s_bar->d+s_bar (0), u_bar+s_bar->u_bar+s_bar (0), 
    # d_bar+s_bar->d_bar+s_bar (0), s_bar+s_bar->s_bar+s_bar (1), g+s_bar->g+s_bar (5)
    # s+s_bar->s+s_bar (2), u+u_bar->s+s_bar (3), d+d_bar->s+s_bar (3), gg->s+s_bar (4)
    
    # ug->ug (5), dg->dg (5), sg->sg (5), u_bar+g->u_bar+g (5), 
    # d_bar+g->d_bar+g (5), s_bar+g->s_bar+g (5), 
    # u+u_bar->gg(4), d+d_bar->gg(4), s+s_bar->gg(4),
    # gg->gg (6)
    
    >>>
    return:
        flavor and collision_type:
            numpy arrays.
            
            # flavor: all possible collisions for the given final particle, eg: 
            #        for final d, we have
            #        ud->ud (0), dd->dd (1), sd->sd (0), u_bar+d->u_bar+d (0), s_bar+d->s_bar+d (0), gd->gd (5)
            #        d_bar+d->d_bar+d (2), u_bar+u->d_bar+d (3), s_bar+s->d_bar+d (3), gg->d_bar+d (4)
            
            The corresponding flavor array is
            #        flavor=np.array([[[0,1,0,1],[1,1,1,1],[2,1,2,1],[3,1,3,1],[5,1,5,1],[6,1,6,1],\
            #                         [4,1,4,1],[3,0,4,1],[5,2,4,1],[6,6,4,1]]],dtype=np.int64)
            
            # collision_type: an index indicate which collision type the process belongs to, eg:
            For final d quark case
            #                collision_type=np.array([[0,1,0,0,0,5,2,3,3,4]],dtype=np.int64)
            
            where 0,1,2,3,4,5,6 corresponds to the following processes:
            (0): q1 + q2 -> q1 + q2
            (1): q + q -> q + q
            (2): q + qbar -> q + qbar
            (3): q1 + q1bar -> q2 + q2bar
            (4): q1 + q1bar -> g + g
            (5): q + g -> q + g
            (6): g + g -> g + g
            
    '''
    particle_order = 'u (0), d (1), s (2), ubar (3), dbar (4), sbar (5), fluon (6)'
    
    flavor, collision_type = {}, {}
    # initialize the values to None
    flavor['2TO2'], flavor['2TO3'], flavor['3TO2'] = None, None, None
    collision_type['2TO2'], collision_type['2TO3'], collision_type['3TO2'] = None, None, None
    
    flavor['2TO2']=np.array([[[0,0,0,0],[1,0,1,0],[2,0,2,0],[4,0,4,0],[5,0,5,0],[6,0,6,0], \
                        [3,0,3,0],[4,1,3,0],[5,2,3,0],[6,6,3,0]], \
                       [[0,1,0,1],[1,1,1,1],[2,1,2,1],[3,1,3,1],[5,1,5,1],[6,1,6,1], \
                        [4,1,4,1],[3,0,4,1],[5,2,4,1],[6,6,4,1]], \
                       [[0,2,0,2],[1,2,1,2],[2,2,2,2],[3,2,3,2],[4,2,4,2],[6,2,6,2], \
                        [5,2,5,2],[3,0,5,2],[4,1,5,2],[6,6,5,2]], \
                       [[1,3,1,3],[2,3,2,3],[3,3,3,3],[4,3,4,3],[5,3,5,3],[6,3,6,3], \
                        [0,3,0,3],[1,4,0,3],[2,5,0,3],[6,6,0,3]], \
                       [[0,4,0,4],[2,4,2,4],[3,4,3,4],[4,4,4,4],[5,4,5,4],[6,4,6,4], \
                        [1,4,1,4],[0,3,1,4],[2,5,1,4],[6,6,1,4]], \
                       [[0,5,0,5],[1,5,1,5],[3,5,3,5],[4,5,4,5],[5,5,5,5],[6,5,6,5], \
                        [2,5,2,5],[0,3,2,5],[1,4,2,5],[6,6,2,5]], \
                       [[0,6,0,6],[1,6,1,6],[2,6,2,6],[3,6,3,6],[4,6,4,6],[5,6,5,6], \
                        [6,6,6,6],[0,3,6,6],[1,4,6,6],[2,5,6,6]]],dtype=np.int32)

    collision_type['2TO2']=np.array([[1,0,0,0,0,5,2,3,3,4], \
                               [0,1,0,0,0,5,2,3,3,4], \
                               [0,0,1,0,0,5,2,3,3,4], \
                               [0,0,1,0,0,5,2,3,3,4], \
                               [0,0,0,1,0,5,2,3,3,4], \
                               [0,0,0,0,1,5,2,3,3,4], \
                               [5,5,5,5,5,5,6,4,4,4]],dtype=np.int32)
        
    return flavor, collision_type, particle_order
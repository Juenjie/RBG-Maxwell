from numba import cuda
import numpy as np
import math, cupy, ray, random

from ..Plasma_single_GPU.main import Plasma_single_GPU
from ..Plasma.utils import find_largest_time_steps
from ..Plasma.utils import check_input_legacy
    
class Plasma():
    def __init__(self, f, dt, \
                 nx_o, ny_o, nz_o, dx, dy, dz, boundary_configuration, \
                 x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                 npx, npy, npz, half_px, half_py, half_pz,\
                 masses, charges, sub_region_relations,\
                 flavor, collision_type, particle_type,\
                 degeneracy, expected_collision_type,\
                 num_gpus_for_each_region,\
                 hbar, c, lambdax, epsilon0, \
                 num_samples = 100, drift_order = 1, rho_J_method="raw",GPU_ids_for_each_region = None):
        '''
        Params
        ======
        f:
            Distribution functions in different spatial regions. Type: dict. 
            The values of the dictionary are numpy arrays.
            e.g. f = {0: numpy array of shape [2,2,1,2,3,4,5,6]}
        dt:
            The infinitesimal time for each time step updation.
            e.g. t = 0.01
        nx_o, ny_o, nz_o: 
            Three lists containing the number of grids used in the x,y and z directions. 
            Each element corresponds to the grid numbers in the indexed region.
            e.g. nx_o = [3, 4]
        dx, dy, dz:
            Infinitesimal differences used in the spatial domain. 
            dx, dy and dz are the same for all sub-regions.
            e.g. dx, dy, dz = 1, 2, 1
        boundary_configuration:
            Dictionary of boundary configurations. Each (key, value) pair stands for the boundary
            configuration of the corresponding region. The values are of tuple types.
            e.g. boundary_configuration = {0: (bound_x, bound_y, bound_z)}
        x_left_bound_o, y_left_bound_o, z_left_bound_o:
            Three lists of the left boundaries in each spatial sub-region.
            e.g. x_left_bound_o = [-1, 0]
        npx, npy, npz:
            Number of momentum grids.
            e.g. npx, npy, npz = 3, 4, 5
        half_px, half_py, half_pz:
            Three lists of the momentum length in x,y and z directions.
            All spatial grids share the same momentum length.
            Each element of half_px stands for the momentum length for each type of
            particles. Hence, len(half_px) = number of particle species.
            e.g. half_px = [2, 3]
        masses:
            List of masses for each particle species.
            e.g. masses = [0.1, 0.1]
        charges:
            List of charges for each particle species.
            e.g. charges = [0, 1]   
        sub_region_relations:
            Dictionary of the relative locations amongest the sub-regions.
            key: 'indicator' gives the index of surfaces to be exchanged.
            key: 'position' gives the relative positions between the regions.
            e.g. sub_region_relations = \
            {'indicator': [[0,3,4],[0,2,4],[0,3,5],[0,2,5],\
                                      [1,3,4],[1,2,4],[1,3,5],[1,2,5]],\
                        'position': [[0,    1,    2,    3,    4,    5,    6,    7],\     -----base
                                     [4,    5,    6,    7,    None, None, None, None],\  -----minus x
                                     [None, None, None, None, 0,    1,    2,    3],\     -----plus x
                                     [None, 0,    None, 2,    None, 4,    None, 6],\     -----minus y
                                     [1,    None, 3,    None, 5,    None, 7,    None],\  -----plus y
                                     [2,    3,    None, None, 6,    7,    None, None],\  -----minus z
                                     [None, None, 0,    1,    None, None, 4,    5]]}     -----plus z
            In the above example, we have 8 sub-regions. Sub-region 0 has three surfaces, i.e.,
            0,3,4 to be exchanged (as stated by the indicator). The first row of 'position'
            gives the index of the 8 sub-regions. The second row gives index of the regions that are 
            to the minus direction of the regions in the first row.
        flavor and collision_type:
            dictionary of 2-2, 2-3, and 3-2 process.
            flavor = {'2TO2:, '2TO3':, ;3TO2':}
            collision_type = {'2TO2:, '2TO3':, ;3TO2':}
            
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

            flavor['2TO3']=np.array([[[0,1,3,4,0], [1,2,1,4,0],               [10001,10001,10001,10001], [10001,10001,10001,10001]],
                                     [[1,2,0,4,1], [10001,10001,10001,10001], [10001,10001,10001,10001], [10001,10001,10001,10001]],
                                     [[2,1,3,4,2], [10001,10001,10001,10001], [10001,10001,10001,10001], [10001,10001,10001,10001]],
                                     [[0,1,0,4,3], [1,2,2,4,3],               [1,4,4,4,3],               [10001,10001,10001,10001]],
                                     [[0,1,0,3,4], [1,2,0,1,4],               [2,1,2,3,4],               [1,4,3,4,4]]],dtype=np.int64)
        
        particle_type:
            particle types correspond to different particles, 0 for classical, 
            1 for fermion, and 2 for bosonic, e.g., [1,1,1]
        degeneracy:
            degenacies for particles, for quarks this is 6, e.g., [6,6,6,6,6,6]
        expected_collision_type:
            e.g., ['2TO2', '2TO3', '3TO2']
        num_gpus_for_each_region:
            Number of GPUs occupied by each spatial region.
            e.g. num_gpus_for_each_region = 0.25
        hbar,c,lambdax,epsilon0:
            numerical value of hbar,c, lambdax and epsilon0 in Flexible Unit (FU)
        num_samples:
            number of samples used in the Monte Carlo integration of the collision term. 
            Recomended value is 100.
        drift_order:
            drift_order for upwind scheme
        rho_J_method:
            str:"quasi_neutral", "raw", etc..
        GPU_ids_for_each_region:
            the list of GPU_ids to be used. If multiple GPUs are available, the first GPU should be better neglected.
            '''
        
        # check if the input is legal
        check_input_legacy(f, dt, \
                           nx_o, ny_o, nz_o, dx, dy, dz, boundary_configuration, \
                           x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                           npx, npy, npz, half_px, half_py, half_pz,\
                           masses, charges,\
                           sub_region_relations,\
                           num_gpus_for_each_region,\
                           num_samples,\
                           flavor, collision_type, particle_type,\
                           degeneracy, expected_collision_type)
        
        number_regions = len(f)
        
        # define ray remote functions
        plasma_single_GPU = (ray.remote(num_gpus = num_gpus_for_each_region))(Plasma_single_GPU)
        
        self.number_regions = len(f)
        self.global_dt = dt
        dpx, dpy, dpz = 2*half_px/npx,2*half_py/npy,2*half_pz/npz
        
        self.cubic_positions = sub_region_relations['position']
        self.bound_indicator = sub_region_relations['indicator']
        
        # For a correct finite difference in spatial regions,
        # all dx (or dy or dz) in different spatial regions must be same.
        # dx, dy and dz can be different
        # However, different values of dx (or dy or dz) are allowed in the calculation
        # of the EM fields. Here, we set the dx values to be the same since a global
        # finite difference method is used in finding the Drift terms.
        dx_o, dy_o, dz_o = [dx]*number_regions, [dy]*number_regions, [dz]*number_regions
   

        # dictionary for saving different regions
        self.plasmas = {}
        for region_id in range(number_regions):
            
            # len_time_snapshots will be used in the calculation of EM fields
            # On each GPU card we store the rho = [rho_t0, rho_t1, ...] and J = [J_t0, J_t1, ...]
            # the length of rho and J are determined by the largest distances between the two sub-regions.
            self.len_time_snapshots = []
            for i_reg_o in range(number_regions):
                # This should be evaluated via find_largest_distances
                distance = find_largest_time_steps(dx, dy, dz, \
                                                   x_left_bound_o[i_reg_o], \
                                                   y_left_bound_o[i_reg_o], \
                                                   z_left_bound_o[i_reg_o], \
                                                   dx, dy, dz, \
                                                   x_left_bound_o[region_id], \
                                                   y_left_bound_o[region_id], \
                                                   z_left_bound_o[region_id], \
                                                   nx_o[i_reg_o], ny_o[i_reg_o], nz_o[i_reg_o], \
                                                   nx_o[region_id], ny_o[region_id], nz_o[region_id],\
                                                   dt, c)
                self.len_time_snapshots.append(distance)
                print(self.len_time_snapshots)

            # initialize different plasma regions
            self.plasmas[region_id] = plasma_single_GPU.remote(region_id, f[region_id], \
                                                               dx, dy, dz, dpx, dpy, dpz, \
                                                               masses, charges, \
                                                               nx_o[region_id], ny_o[region_id], nz_o[region_id], \
                                                               npx, npy, npz, \
                                                               x_left_bound_o[region_id], \
                                                               y_left_bound_o[region_id], \
                                                               z_left_bound_o[region_id], \
                                                               half_px, half_py, half_pz, \
                                                               number_regions, dt,\
                                                               self.len_time_snapshots, \
                                                               dx_o, dy_o, dz_o,\
                                                               x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                                                               nx_o, ny_o, nz_o,\
                                                               degeneracy, particle_type,\
                                                               boundary_configuration[region_id][0], \
                                                               boundary_configuration[region_id][1],\
                                                               boundary_configuration[region_id][2],\
                                                               expected_collision_type, num_samples, \
                                                               sub_region_relations['indicator'][region_id],\
                                                               flavor, collision_type,\
                                                               drift_order = drift_order, \
                                                               hbar = hbar, c = c, \
                                                               lambdax = lambdax, epsilon0 = epsilon0, rho_J_method = rho_J_method, GPU_id = GPU_ids_for_each_region[region_id])
            
    def exchange_boundaries(self):
        '''
        Exchange the boundaries of the overlapped regions.
        All values in the x-direction are exchanged, and then the y-deirection and z-direction.
        '''
        
        for i_reg in range(self.number_regions):   
            # reshape to 6-d
            self.plasmas[i_reg].reshape_to_six_dim.remote()
                    
        for region_id in range(self.number_regions):    
            # replace the values in minus x direction
            if self.cubic_positions[1][region_id] != None:
                if 0 not in self.bound_indicator[region_id]:
                    raise AssertionError("The minus x boundary of Region {} is incorrectly set. Current region id: {}, the exchanging surfaces of this region: {}. Region {} is not configured to be next to region {}."\
                                         .format(region_id,region_id,self.bound_indicator[region_id],self.cubic_positions[1][region_id],region_id)) 
                else:
                    surface = ray.get(self.plasmas[region_id].take_boundary.remote(0))
                    self.plasmas[self.cubic_positions[1][region_id]].reset_boundary.remote(surface, 1)

            # replace the values in plus x direction
            if self.cubic_positions[2][region_id] != None:
                if 1 not in self.bound_indicator[region_id]:
                    raise AssertionError("The plus x boundary of Region {} is incorrectly set. Current region id: {}, the exchanging surfaces of this region: {}. Region {} is not configured to be next to region {}."\
                                         .format(region_id,region_id,self.bound_indicator[region_id],self.cubic_positions[2][region_id],region_id)) 
                else:
                    surface = ray.get(self.plasmas[region_id].take_boundary.remote(1))
                    self.plasmas[self.cubic_positions[2][region_id]].reset_boundary.remote(surface, 0)

        for region_id in range(self.number_regions):    
            # replace the values in minus y direction
            if self.cubic_positions[3][region_id] != None:
                if 2 not in self.bound_indicator[region_id]:
                    raise AssertionError("The minus y boundary of Region {} is incorrectly set. Current region id: {}, the exchanging surfaces of this region: {}. Region {} is not configured to be next to region {}."\
                                         .format(region_id,region_id,self.bound_indicator[region_id],self.cubic_positions[3][region_id],region_id))  
                else:
                    surface = ray.get(self.plasmas[region_id].take_boundary.remote(2))
                    self.plasmas[self.cubic_positions[3][region_id]].reset_boundary.remote(surface, 3)

            # replace the values in plus y direction
            if self.cubic_positions[4][region_id] != None:
                if 3 not in self.bound_indicator[region_id]:
                    raise AssertionError("The plus y boundary of Region {} is incorrectly set. Current region id: {}, the exchanging surfaces of this region: {}. Region {} is not configured to be next to region {}."\
                                         .format(region_id,region_id,self.bound_indicator[region_id],self.cubic_positions[4][region_id],region_id))  
                else:
                    surface = ray.get(self.plasmas[region_id].take_boundary.remote(3))
                    self.plasmas[self.cubic_positions[4][region_id]].reset_boundary.remote(surface, 2)

        for region_id in range(self.number_regions):        
            # replace the values in minus z direction
            if self.cubic_positions[5][region_id] != None:
                if 4 not in self.bound_indicator[region_id]:
                    raise AssertionError("The minus z boundary of Region {} is incorrectly set. Current region id: {}, the exchanging surfaces of this region: {}. Region {} is not configured to be next to region {}."\
                                         .format(region_id,region_id,self.bound_indicator[region_id],self.cubic_positions[5][region_id],region_id))  
                else:
                    surface = ray.get(self.plasmas[region_id].take_boundary.remote(4))
                    self.plasmas[self.cubic_positions[5][region_id]].reset_boundary.remote(surface, 5)

            # replace the values in plus z direction
            if self.cubic_positions[6][region_id] != None:
                if 5 not in self.bound_indicator[region_id]:
                    raise AssertionError("The plus z boundary of Region {} is incorrectly set. Current region id: {}, the exchanging surfaces of this region: {}. Region {} is not configured to be next to region {}."\
                                         .format(region_id,region_id,self.bound_indicator[region_id],self.cubic_positions[6][region_id],region_id)) 
                else:
                    surface = ray.get(self.plasmas[region_id].take_boundary.remote(5))
                    self.plasmas[self.cubic_positions[6][region_id]].reset_boundary.remote(surface, 4)
        
        for i_reg in range(self.number_regions):   
            # reshape to 1-d
            self.plasmas[i_reg].reshape_to_one_dim.remote()
            
    def proceed_one_step(self, i_time, total_time_steps, processes = {'VT':1., 'DT':1., 'CT':1.}, \
                         BEx = 0., BEy = 0., BEz = 0., BBx = 0., BBy = 0., BBz = 0.):
        '''
        proceed time updation. This involves the following sub-stages:
        1. Try evaluate all regions and return the suggested dt
        2. Accept the suggestion and perform time straitification
            1. electric \rho and J are calculated via the distribution function f. 
            2. B and E are calculated from the Jefimenko's equations.
            3. External EM forces and other forces are obtained.
            4. Obtain the Vlasov and (causal) Drift terms.
            5. Collision terms at all levels are obtained.
            6. Update the distribution f(t+1) = dt*(f(t) - Vt - Dt + Ct) with time stratification method.
            7. Exchange the distributions amongest the momentum levels
        3. Exchange the boundaries
        4. Exchange electromagnetci fields
        
        Params
        ======
        i_time:
            the i-th time step
        total_time_steps:
            the total time steps
        processes:
            the processes that need to be considered, default values are 
            'VT', 'DT', and 'CT' indicating Vlasov term, Drift term and Collision term.
        BEx, BEy, BEz, BBx, BBy, BBz： 
            numpy arrays correspond to the background EM fields
        '''
        whether_terminate = []
        # evaluate the distribution function in all regions
        for i_reg in range(self.number_regions):
            whether_terminate.append(self.plasmas[i_reg].proceed_one_step.remote(i_time, total_time_steps, self.global_dt, try_evaluate = True, processes = processes, BEx = BEx[i_reg], BEy = BEy[i_reg], BEz = BEz[i_reg], BBx = BBx[i_reg], BBy = BBy[i_reg], BBz = BBz[i_reg]))
            
        for i_reg in range(self.number_regions):
            whether_terminate[i_reg] = ray.get(whether_terminate[i_reg])

        # terminate if illlegal results have been obtained
        if any(whether_terminate) == True:
            raise AssertionError("The code has been terminated since the obtained distribution is unphysical!") 
        
        # exchange the boundaries after evaluation
        if self.number_regions > 1:
            self.exchange_boundaries()
            
        # exchange EM fields for Vlasov term
        if processes['VT']>0.5:
            self.exchange_EM_fields()

        # load stream_ticker
        # stream_ticker transfers a small data from GPU to CPU
        # this gives a signal to the GPU that whether each time step is finished 
        self.acquire_values("stream_ticker")
        
        '''# This is decrapted since a UPFD has been adopted.
        # try evaluate in all regions
        # collect count and dt to see if any of the regions are stratified
        scale_collection = []
        for i_reg in range(self.number_regions):
            # collect count and dt in all regions
            scale_collection.append(ray.get(self.plasmas[i_reg].proceed_one_step.remote(i_time, total_time_steps, self.global_dt, try_evaluate = True, scale = 0, \
                           processes = processes,\
                           BEx = BEx[i_reg], BEy = BEy[i_reg], BEz = BEz[i_reg], BBx = BBx[i_reg], BBy = BBy[i_reg], BBz = BBz[i_reg])))   

        # if all regions are not stratified, then accept the results and exchange boundaries
        if sum(scale_collection) < 1:  # the minimum scale is 5 if need to rescale time step
            # accept the updation
            for i_reg in range(self.number_regions):   
                # accetp the obtained distribution
                self.plasmas[i_reg].accept_fp.remote()
            # exchange the boundaries
            if self.number_regions > 1:
                self.exchange_boundaries()
                
        # if stratification is required
        else:
            # find the largest scale and proceed straitification
            max_scale = max(scale_collection)
     
            # keep collision term unchanged while loop
            processes['CT'] = 0.
            
            for i_sub_time in range(max_scale):
                for i_reg in range(self.number_regions):
                    # obtain distribution in all regions with global_dt/max_scale
                    # the new distribution will be accepted automatically
                    self.plasmas[i_reg].proceed_one_step.remote(i_time, total_time_steps, \
                           self.global_dt/max_scale, try_evaluate = False, scale = max_scale, \
                           processes = processes,\
                           BEx = BEx[i_reg], BEy = BEy[i_reg], BEz = BEz[i_reg], BBx = BBx[i_reg], BBy = BBy[i_reg], BBz = BBz[i_reg])
                # exchange the boundaries at each sub-time  
                if self.number_regions > 1:
                    self.exchange_boundaries()'''


    def acquire_values(self, quantity):
        '''
        return the values of all regions.
        
        params
        ======
        quantity:
            string, can be:   'Vlasov term','Drift term', 'Collision term', 'Distribution',
                              'Forces', "Electric rho/J", 
                              "EM fields on current region",
                              "EM fields to other regions", "number rho/J", "stream_ticker"
        retrun
        ======
        quantities_required:
            dictionary, (key, value) corresponds to (region_id, quantities).'''
        
        quantities_required = {}
        for i_reg in range(self.number_regions):   
            quantities_required[i_reg] = ray.get(self.plasmas[i_reg].return_self.remote(quantity))
          
        return quantities_required
    
    def exchange_EM_fields(self):
        '''
        In each region, Ex_dis, Ey_dis, Ez_dis, Bx_dis, By_dis, Bz_dis stores the electromagnetic fields
        generated by that region to other regions.'''
        
        # For each region, first take the electromagnetic fields
        for i_sender in range(self.number_regions):
            Ex_dis, Ey_dis, Ez_dis, Bx_dis, By_dis, Bz_dis = \
            ray.get(self.plasmas[i_sender].return_self.remote("EM fields to other regions"))
            
            # send these electromagnetic fields to other regions
            for i_receiver in range(self.number_regions):
                self.plasmas[i_receiver].update_EM_fields.remote(Ex_dis[i_receiver], Ey_dis[i_receiver], \
                                                                 Ez_dis[i_receiver], Bx_dis[i_receiver], \
                                                                 By_dis[i_receiver], Bz_dis[i_receiver], \
                                                                 i_sender)
                    
                    
                    
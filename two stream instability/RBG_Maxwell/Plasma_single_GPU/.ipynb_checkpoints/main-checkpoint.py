from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np
import math, cupy, ray, random, os
import copy

from ..Collision_term.main import Collision_term
from ..External_forces.main import External_forces
from ..Vlasov_Drift_terms.main import Drift_Vlasov_terms
from ..EMsolver.solver import EMsolver
from ..EMsolver.region_distance import signal_indicator
from ..Macro_quantities.main import density, charged_density   
from ..Plasma_methods.main import plasma_method
from ..Plasma_single_GPU.utils import check_legacy_of_distributions

def boundary_coding(nx, ny, nz, npx, npy, npz, \
                    number_momentum_levels, num_particle_species, \
                    bound_indicator):
    '''
    Give the index of the boundaries for surfaces, edges and corners.
    Also give the dictionary of containing the values of the boundaries
    Params
    ======
    ny, ny, nz, npx, npy, npz:
        the number of phase space grids
    number_momentum_levels:
        number of momentum levels
    num_particle_species:
        number of particle species
    bound_indicator:
        a list containing the surfaces, edges and corners. e.g.[0,1,2,3,4,5]
        these values correspond to
        [-x,+x,-y,+y,-z,+z]
        if a boundary is not to be exchanged, omit it in bound_indicator 
        Here -x means the -x boundary surface will be exchanged, indicating there is
        a box next to the -x boundary.
    '''

    # these are the boundaries that need to be taken
    surface_index_taken = np.array([[1,    2,    0,    ny,   0,    nz],\
                                    [nx-2, nx-1, 0,    ny,   0,    nz],\
                                    [0,    nx,   1,    2,    0,    nz],\
                                    [0,    nx,   ny-2, ny-1, 0,    nz],\
                                    [0,    nx,   0,    ny,   1,    2],\
                                    [0,    nx,   0,    ny,   nz-2, nz-1]])

    # these are the boundaries that need to be exchanged
    surface_index = np.array([[0,    1,    0,    ny, 0,    nz],\
                              [nx-1, nx,   0,    ny, 0,    nz],\
                              [0,    nx,   0,    1,  0,    nz],\
                              [0,    nx,   ny-1, ny, 0,    nz],\
                              [0,    nx,   0,    ny, 0,    1],\
                              [0,    nx,   0,    ny, nz-1, nz]])
    
    return surface_index_taken, surface_index

class Plasma_single_GPU():
    def __init__(self, region_id, f, dx, dy, dz, dpx, dpy, dpz, masses, charges,\
                 nx, ny, nz, npx, npy, npz, x_left_bound, y_left_bound, z_left_bound, \
                 half_px, half_py, half_pz, number_regions, dt,\
                 len_time_snapshots, dx_o, dy_o, dz_o,\
                 x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                 nx_o, ny_o, nz_o, degeneracy, particle_type,
                 x_bound_config, y_bound_config, z_bound_config, \
                 expected_collision_type, num_samples = 100, bound_indicator = None,\
                 flavor = None, collision_type = None, drift_order = 1, \
                 hbar = None, c = None, lambdax = None, epsilon0 = None, rho_J_method = "raw", GPU_id = '0' ):
        
        """Main class to perform the plasma evolution on a single GPU card.
        ======
        Params:
        region_id:
            the index of the current spatial region, e.g., 1
        f: 
            distribution function for all particles at all levels. The shape of f corresponds to
            [momentum_levels, particle_species, nx*ny*nz*npx*npy*npz].
        dx, dy, dz: 
            infinitesimal difference in spatial domain
        dpx, dpy, dpz: 
            infinitesimal difference in momentum domain
            this is an array containing the values for different type of particles
        masses: 
            the mass of the particles, cp-array, the order should follow the particles in distribution function
        charges: 
            the charge of the particles, cp-array, the order should follow the particles in distribution function
        nx, ny, nz: 
            number of spatial grids 
        npx, npy, npz: 
            number of momentum grids
        x_left_bound, y_left_bound, z_left_bound: 
            the value of the left boundaries of spatial domain
        half_px, half_py, half_pz: 
            the value of half momentum domain
            this is an array containing the values for different type of particles 
        number_regions: 
            total number of spatial regions
        dt: 
            time step
        ############################################################################
        These parameters are for the EMsolver
        len_time_snapshots: 
            how many time steps of rho and J to be saved on GPU memory
        dx_o, dy_o, dz_o: 
            three lists containing the infinitesimals of spatial domains on each GPU
        x_left_boundary_o, y_left_boundary_o, z_left_boundary_o: 
            three lists containing the left boundary values
            of spatial domains on each GPU
        nx_o, ny_o, nz_o: 
            three lists containing the number of grids on each GPU
        #############################################################################
        degeneracy:
            particle degeneracy, e.g., [6,16] for quark and gluon
        particle_type:
            0,1,2 stand for claassical, fermion, and boson particles
        x_bound_config, y_bound_config, z_bound_config:
            configuretions of the boundary conditions. 
            x_bound_config is of shape [ny, nz, 2]
            y_bound_config is of shape [nx, nz, 2]
            z_bound_config is of shape [nx, ny, 2]
            the last two components denote:
            alpha --- the component being reflected
            beta --- the component of the momentum in use for the reflected particles
        expected_collision_type:
            expected collision type, list, e.g., ['2TO2', '2TO3', '3TO2']
        num_samples:
            number of samples in the Monte Carlo integration of the collision term
        bound_indicator:
            a list containing the surfaces, edges and corners. e.g.[0,1,2,3,4,5]
            these values correspond to
            [-x,+x,-y,+y,-z,+z]
            if a boundary is not to be exchanged, omit it in bound_indicator 
            Here -x means the -x boundary surface will be exchanged, indicating there is
            a box next to the -x boundary. 
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
        
        drift_order:
            order of drift terms in upwind scheme
        hbar,c,lambdax,epsilon0:
            numerical value of hbar,c, lambdax and epsilon0 in Flexible Unit (FU)        
        rho_J_method:
            str:"quasi_neutral", "raw", etc..
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
        # initialize the parameters across the methods.        
        self.region_id = region_id
        self.number_regions = number_regions    
        
        # parameters accross the methods
        # grid related
        self.nx, self.ny, self.nz = nx, ny, nz
        self.npx, self.npy, self.npz = npx, npy, npz        
        self.total_phase_grids = nx*ny*nz*npx*npy*npz
        self.total_spatial_grids = nx*ny*nz
        self.total_momentum_grids = npx*npy*npz
        self.middle_npx, self.middle_npy, self.middle_npz = int(npx/2), int(npy/2), int(npz/2)
        self.nx_o, self.ny_o, self.nz_o = nx_o, ny_o, nz_o # array, for different spatial region
        self.drift_order = drift_order
        
        # boundary values related 
        self.x_left_bound, self.y_left_bound, self.z_left_bound = x_left_bound, y_left_bound, z_left_bound
        self.half_px, self.half_py, self.half_pz = cupy.asarray(half_px), cupy.asarray(half_py), cupy.asarray(half_pz)
        self.x_left_bound_o, self.y_left_bound_o, self.z_left_bound_o = \
            x_left_bound_o, y_left_bound_o, z_left_bound_o
        self.x_bound_config = cupy.asarray(x_bound_config)
        self.y_bound_config = cupy.asarray(y_bound_config)
        self.z_bound_config = cupy.asarray(z_bound_config)
 
        # infinitesimal related
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dpx, self.dpy, self.dpz = cupy.asarray(dpx), cupy.asarray(dpy), cupy.asarray(dpz) # array, for different types
        self.dp = cupy.asarray(dpx*dpy*dpz) # array, for different types
        self.momentum_volume_element = cupy.asarray(dpx*dpy*dpz) # this is an array containing the values for different type of particles

        self.dx_o, self.dy_o, self.dz_o = dx_o, dy_o, dz_o # array for different spatial regions
        self.dt = dt
        
        # physics related
        self.masses = cupy.asarray(masses)
        self.charges = cupy.asarray(charges)
        self.rho_J_method = rho_J_method
        if rho_J_method == 'raw':
            self.quasi_neutral = 0
        elif rho_J_method == 'quasi_neutral':
            self.quasi_neutral = 1
        else:
            raise AssertionError("Plasma method must be raw or quasi_neutral!")      
        
        
        # # flavor and collision_type for various collisions
        self.flavor, self.collision_type = {}, {}
        if '2TO2' in expected_collision_type:
            self.flavor['2TO2'] = cupy.asarray(flavor['2TO2'])
            self.collision_type['2TO2'] = cupy.asarray(collision_type['2TO2'])
        if '2TO3' in expected_collision_type:
            self.flavor['2TO3'] = cupy.asarray(flavor['2TO3'])
            self.collision_type['2TO3'] = cupy.asarray(collision_type['2TO3'])
        if '3TO2' in expected_collision_type:
            self.flavor['3TO2'] = cupy.asarray(flavor['3TO2'])
            self.collision_type['3TO2'] = cupy.asarray(collision_type['3TO2'])
        self.allowed_collision_type = expected_collision_type

        self.hbar, self.c,self.lambdax,self.epsilon0 = hbar, c, lambdax, epsilon0
        
        # sizes of the relavant arrays        
        self.num_particle_species = len(masses)
        self.degeneracy = cupy.asarray(degeneracy)
        self.particle_type = cupy.asarray(particle_type)
        self.number_momentum_levels = len(f)
        self.num_samples = num_samples
        self.len_time_snapshots = len_time_snapshots
        
        '''
#         # scanning of the momentum
#         unit_f = cupy.ones([self.nx, self.ny, self.nz, self.npx, self.npy, self.npz])
#         self.scanned_px, self.scanned_py, self.scanned_pz = [], [], []
#         for i_type in range(self.num_particle_species):
#             px = (cupy.arange(self.npx)+0.5)*self.dpx[i_type]-self.half_px[i_type]
#             py = (cupy.arange(self.npy)+0.5)*self.dpy[i_type]-self.half_py[i_type]
#             pz = (cupy.arange(self.npz)+0.5)*self.dpz[i_type]-self.half_pz[i_type]
            
#             self.scanned_px.append(cupy.multiply(unit_f.swapaxes(0,-3), px).swapaxes(0,-3))
#             self.scanned_py.append(cupy.multiply(unit_f.swapaxes(0,-2), py).swapaxes(0,-2))
#             self.scanned_pz.append(cupy.multiply(unit_f.swapaxes(0,-1), pz).swapaxes(0,-1))

#         self.scanned_p = []
#         for i_type in range(self.num_particle_species):
#             # cupy.sqrt(px**2+py**2+pz**2+m**2*c**2)
#             self.scanned_p.append(cupy.sqrt(px**2 + py**2 + pz**2 +\
#                                             self.masses[i_type]**2*self.c**2))
              
#         # give vx, vy, vz grid
#         self.scanned_vx, self.scanned_vy, self.scanned_vz = [], [], []
#         for i_type in range(self.num_particle_species):
#             self.scanned_vx.append(self.c*self.scanned_px[i_type]/self.scanned_p[i_type])
#             self.scanned_vy.append(self.c*self.scanned_py[i_type]/self.scanned_p[i_type])
#             self.scanned_vz.append(self.c*self.scanned_pz[i_type]/self.scanned_p[i_type])
#         print(self.scanned_vx, self.scanned_vz)
        '''
        
        # Configure the blocks
        self.threadsperblock = 32
        # configure the grids
        self.blockspergrid_total_phase = (self.total_phase_grids + (self.threadsperblock - 1)) // self.threadsperblock
        self.blockspergrid_spatial = (self.total_spatial_grids + (self.threadsperblock - 1)) // self.threadsperblock
        
        # arrays on GPUs
        self.f = cupy.asarray(f)   
        
        # seeds for random generator
        self.rng_states = cuda.to_device\
                          (create_xoroshiro128p_states\
                           (self.blockspergrid_total_phase*self.threadsperblock,
                            seed=random.sample(range(0,1000),1)[0]))
        
        # store EM fields at the current GPU region that are from other GPUs
        # each row of Ex, stands for the EM fields from other GPUs
        self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz = \
                        (cupy.zeros([self.number_regions, self.nx*self.ny*self.nz]) for _ in range(6))  

        # initialize EM_solver, the results will be handled to all GPUs
        self.EMsolver = []
        # on CPU
        self.Ex_dis, self.Ey_dis, self.Ez_dis, self.Bx_dis, self.By_dis, self.Bz_dis = {}, {}, {}, {}, {}, {}

        for i_reg in range(number_regions):
            # index for rho/J in generating EM
            self.id_xl, self.id_xr, self.id_yl, self.id_yr, self.id_zl, self.id_zr = 0, self.nx, 0, self.ny, 0, self.nz
#             print('index')
#             print(self.id_xl, self.id_xr, self.id_yl, self.id_yr, self.id_zl, self.id_zr)
            # the source region needs to be shrinked, according to the ghost boundaries
            self.x_left_bound_adj, self.y_left_bound_adj, self.z_left_bound_adj = self.x_left_bound, self.y_left_bound, self.z_left_bound
            self.nx_adj, self.ny_adj, self.nz_adj = self.nx, self.ny, self.nz
        
            # check the -x, -y, -z, +x, +y, +z directions           
            if 0 in bound_indicator:
                self.nx_adj -= 1
                self.x_left_bound_adj += self.dx
                self.id_xl += 1
            if 1 in bound_indicator:
                self.nx_adj -= 1
                self.id_xr -= 1
            if 2 in bound_indicator:
                self.ny_adj -= 1
                self.y_left_bound_adj += self.dy
                self.id_yl += 1
            if 3 in bound_indicator:
                self.ny_adj -= 1
                self.id_yr -= 1
            if 4 in bound_indicator:
                self.nz_adj -= 1
                self.z_left_bound_adj += self.dz
                self.id_zl += 1
            if 5 in bound_indicator:
                self.nz_adj -= 1  
                self.id_zr -= 1
#             print('index after')
#             print(self.id_xl, self.id_xr, self.id_yl, self.id_yr, self.id_zl, self.id_zr)
            # initialize the EM solver
            # the source region needs to be shrinked, according to the ghost boundaries
            self.EMsolver.append(EMsolver(self.len_time_snapshots[i_reg], \
                                          self.nx_o[i_reg], self.ny_o[i_reg], self.nz_o[i_reg], \
                                          self.nx_adj, self.ny_adj, self.nz_adj,\
                                          self.dx_o[i_reg], self.dy_o[i_reg], self.dz_o[i_reg], \
                                          self.x_left_bound_o[i_reg], self.y_left_bound_o[i_reg], \
                                          self.z_left_bound_o[i_reg], \
                                          self.dx, self.dy, self.dz, \
                                          self.x_left_bound_adj, self.y_left_bound_adj, self.z_left_bound_adj, \
                                          self.dt, self.epsilon0, self.c)) 
            # electro-magnetic fields generated by the current region to all other regions 
            self.Ex_dis[i_reg], self.Ey_dis[i_reg], self.Ez_dis[i_reg], \
            self.Bx_dis[i_reg], self.By_dis[i_reg], self.Bz_dis[i_reg] = \
            (np.zeros([self.nx_o[i_reg]*self.ny_o[i_reg]*self.nz_o[i_reg]]) for _ in range(6))
        
        # give the boundary info
        # these containing the boundaries to be taken (i.e., the boundaries to be used by others)
        # the boundaries index to be taken and to be replaced
        self.surface_index_taken, self.surface_index = \
        boundary_coding(self.nx, self.ny, self.nz, self.npx, self.npy, self.npz, \
                        self.number_momentum_levels, self.num_particle_species, \
                        bound_indicator)

        print("Using context {} for region {}.".format(cuda.current_context().device.uuid, region_id))
        
        # load stream_ticker
        # stream_ticker transfers a small data from GPU to CPU
        # this gives a signal to the GPU that whether each time step is finished.
        self.stream_ticker = cupy.array([0])

    def proceed_one_step(self, current_time_step, total_time_steps, \
                         dt, try_evaluate, \
                         processes = {'VT':1., 'DT':1., 'CT':1.},\
                         BEx = 0., BEy = 0., BEz = 0., BBx = 0., BBy = 0., BBz = 0.):
        """
        Exchange the boundaries before proceed!!!
        Proceed one time step of the plasma evolution.
        The evolution includes the following sublines:
        1. electric \rho and J are calculated via the distribution function f. 
        2. B and E are calculated from the Jefimenko's equations.
        3. External EM forces and other forces are obtained.
        4. Obtain the Vlasov and (causal) Drift terms.
        5. Collision terms at all levels are obtained.
        6. Update the distribution f(t+1) = dt*(f(t) - Vt - Dt + Ct) with time stratification method.
        7. Exchange the distributions amongest the momentum levels
        
        params
        ======
        processes:
            the processes that need to be considered, default values are 
            'VT', 'DT', and 'CT' indicating Vlasov term, Drift term and Collision term.
        BEx, BEy, BEz, BBx, BBy, BBz： 
            numpy arrays correspond to the background EM fields
        """  
        
        # sending the background EM fields to GPU
        BEx, BEy, BEz, BBx, BBy, BBz = cupy.asarray(BEx), cupy.asarray(BEy), cupy.asarray(BEz),\
                                       cupy.asarray(BBx), cupy.asarray(BBy), cupy.asarray(BBz)

        # evalute new distribution
        whether_terminate = self.PF_PT(processes, BEx, BEy, BEz, BBx, BBy, BBz, \
                             dt, current_time_step, total_time_steps)
        return whether_terminate
        '''# This is decrapted since a UPFD has been adopted.
        if try_evaluate == True: 
            # sending the background EM fields to GPU
            BEx, BEy, BEz, BBx, BBy, BBz = cupy.asarray(BEx), cupy.asarray(BEy), cupy.asarray(BEz),\
                                           cupy.asarray(BBx), cupy.asarray(BBy), cupy.asarray(BBz)
        
            # evalute new distribution
            self.PF_PT(processes, BEx, BEy, BEz, BBx, BBy, BBz, \
                       dt, current_time_step, total_time_steps, eval_External_Forces = True)

            # check is the obtained self.fp is legal.
            whether_legal, illegal_type, value, which_species = check_legacy_of_distributions(self.fp, self.particle_type, self.hbar)
            
            if whether_legal == False:
                
                However, we claim here that this procedure is not perfect, even though we have solved the 
                illegal distributions in try_evaluate, we cannot do so here.
                This is because if any of the steps here need to be stritified, the entire execution time will
                become inacceptable. Therefore, we assume that all distributions generated in the smaller steps
                are always legal (legal means f>0 and f_fermion<1).
                
                # find the proper scale
                for scale in [5, 10, 20, 50, 100, 300, 1000]:
                    # try evalute new distribution while keep collision term and EM fields fixed.
                    self.DT_VT(self.f, dt/scale, current_time_step, processes)
                    # check is the obtained self.fp is legal.
                    whether_legal_scale, _, _, _ = check_legacy_of_distributions(self.fp, self.particle_type, self.hbar)
                    # return scale if a new scale is found
                    if whether_legal_scale == True:
                        print('need to rescale at time step ', current_time_step,'for region ',self.region_id,', since the ', illegal_type,' of species: ', which_species, ' is ',value,'.')
                        return scale
                    
                    # assert error if no scale is found
                    if scale == 1000:
                        raise AssertionError("The algorithm fails to give correct distribution with. Try to choose a proper and smooth initial distribution.") 
            else:
                # return the value of scale
                return scale
        else:                 
            # evalute new distribution while keep collision term and EM fields fixed.
            self.DT_VT(self.f, dt, current_time_step, processes)
            self.accept_fp()'''
             
    
    def accept_fp(self):
        '''Simply copy the values of self.fp to self.f
        This is used when a trial in proceed_one_step is accepted'''
        self.f = cupy.copy(self.fp)
        
    def rho_J_to_EB(self):
        '''
        This method gives the electro-magnetic fields according to the given rho and J.
        '''
        
        # Obtain electric \rho and J from the distribution f
        # electric_rho/electric_Jx/electric_Jy/electric_Jz: [nx*ny*nz]
        
        self.electric_rho, self.electric_Jx, self.electric_Jy, self.electric_Jz = \
        charged_density(self.f, self.num_particle_species, self.total_spatial_grids, \
                        self.masses, self.charges, \
                        self.total_phase_grids, self.momentum_volume_element, self.npx, self.npy, self.npz, \
                        self.nx, self.ny, self.nz, self.half_px, self.half_py, self.half_pz, \
                        self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz, self.number_momentum_levels,\
                        self.blockspergrid_total_phase, self.threadsperblock, self.c)
        
#         if 1000 > self.current_time_step:
#             print('self.f.max(), self.charges, self.dx, self.dy, self.dz, self.electric_rho.min, max')
#             print(self.f.max(), self.charges, self.dx, self.dy, self.dz, self.electric_rho.min(), self.electric_rho.max())
            
            
        # Here we use different method to obtain EM fields
        if self.rho_J_method == "raw":            
            # Calculate B and E at all GPU cards from the Jefimenko's equations
            # note that these EM fields are for each GPU card , and NOT on this card
            # Here, it is the EM fields on all GPU cards induced by the rho and J on this card
            # The obtained EM fields will be distributed to other GPUs
            
            
#             print('self.number_regions')
#             print(self.number_regions)
            # dis here stands for to be distributed
            for i_reg in range(self.number_regions):
#                 print('okkkkkkkkkkkkkkkkkk')
#                 print('self.EMsolver')
#                 print(len(self.EMsolver))
#                 print('input rho and J in main')
#                 print(self.electric_rho, self.electric_Jx, self.electric_Jy, self.electric_Jz)
#                 print('other input info')
#                 print(self.f.sum(),self.f.shape, self.num_particle_species, self.total_spatial_grids, \
#                         self.masses, self.charges, \
#                         self.total_phase_grids, self.momentum_volume_element, self.npx, self.npy, self.npz, \
#                         self.nx, self.ny, self.nz, self.half_px, self.half_py, self.half_pz, \
#                         self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz, self.number_momentum_levels,\
#                         self.blockspergrid_total_phase, self.threadsperblock, self.c)
#                 print('shapes')
#                 print(self.electric_rho.shape, self.electric_Jx.shape, self.electric_Jy.shape, self.electric_Jz.shape)
#                 print(self.nx, self.ny, self.nz, self.id_xl, self.id_xr, self.id_yl, self.id_yr,self.id_zl, self.id_zr)
#                 print('input rho and J in main reshaped')
#                 print(self.electric_rho.reshape([self.nx, self.ny, self.nz]),self.electric_rho.reshape([self.nx, self.ny, self.nz]).shape)
                # evaluate the EM fields
                # the source region needs to be shrinked, according to the ghost boundaries
                self.EMsolver[i_reg].Jefimenko_solver(self.electric_rho.reshape([self.nx, self.ny, self.nz])[self.id_xl:self.id_xr, self.id_yl:self.id_yr,self.id_zl:self.id_zr].flatten(),self.electric_Jx.reshape([self.nx, self.ny, self.nz])[self.id_xl:self.id_xr, self.id_yl:self.id_yr,self.id_zl:self.id_zr].flatten(),self.electric_Jy.reshape([self.nx, self.ny, self.nz])[self.id_xl:self.id_xr, self.id_yl:self.id_yr,self.id_zl:self.id_zr].flatten(),self.electric_Jz.reshape([self.nx, self.ny, self.nz])[self.id_xl:self.id_xr, self.id_yl:self.id_yr,self.id_zl:self.id_zr].flatten(), self.quasi_neutral)
                
                
                self.Ex_dis[i_reg], self.Ey_dis[i_reg], self.Ez_dis[i_reg], \
                self.Bx_dis[i_reg], self.By_dis[i_reg], self.Bz_dis[i_reg] = \
                    self.EMsolver[i_reg].GEx.get(), self.EMsolver[i_reg].GEy.get(), self.EMsolver[i_reg].GEz.get(),\
                    self.EMsolver[i_reg].GBx.get(), self.EMsolver[i_reg].GBy.get(), self.EMsolver[i_reg].GBz.get()
            
        
        elif self.rho_J_method == 'quasi_neutral':
            pass
        else:
            raise AssertionError("Plasma method must be raw or quasi_neutral!")

    def external_forces(self, BEx, BEy, BEz, BBx, BBy, BBz):
        """This method gives the external forces.
        Params
        ======
        BEx, BEy, BEz, BBx, BBy, BBz:
            background EM fields, will be added to the medium generated EM fields
        """
            
        # external forces, these valeus are stored on the GPU memory
        # Fx/Fy/Fz: [momentum_level, particle_species, nx*ny*nz*npx*npy*npz]
        # BEx, BEy, BEz, BBx, BBy, BBz are the background fields
        # the values for self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz will 
        # only be used by the GPU allocators and not on this card
   
        self.Ex_total, self.Ey_total, self.Ez_total, self.Bx_total, self.By_total, self.Bz_total = self.Ex.sum(axis=0)+BEx, self.Ey.sum(axis=0)+BEy, self.Ez.sum(axis=0)+BEz, \
                        self.Bx.sum(axis=0)+BBx, self.By.sum(axis=0)+BBy, self.Bz.sum(axis=0)+BBz
        
        self.Fx, self.Fy, self.Fz = \
        External_forces(self.masses, self.charges,\
                        self.total_phase_grids, self.num_particle_species, \
                        self.npx,self.npy,self.npz,self.nx,self.ny,self.nz, \
                        self.half_px, self.half_py, self.half_pz, \
                        self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz, \
                        self.blockspergrid_total_phase, self.threadsperblock, self.number_momentum_levels,\
                        self.Ex_total, self.Ey_total, self.Ez_total, self.Bx_total, self.By_total, self.Bz_total, self.c)
        # print(self.By_total.sum(), self.Bz_total.sum(),cupy.abs(self.Fx).sum(),cupy.abs(self.Fy).sum(),cupy.abs(self.Fz).sum())
    def DT_VT(self, f_local, dt, current_time_step, processes):
        '''Obtain the drift and vlasov terms'''
        # self.fp is the updated distribution
        # this will be acceppted only after legacy check
        self.f = Drift_Vlasov_terms(f_local, self.Fx, self.Fy, self.Fz, \
                                     self.masses, self.total_phase_grids, self.num_particle_species, \
                                     self.npx, self.npy, self.npz, self.nx, self.ny, self.nz,\
                                     self.half_px, self.half_py, self.half_pz, \
                                     self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz, self.number_momentum_levels,\
                               self.x_bound_config, self.y_bound_config, self.z_bound_config, \
                               self.blockspergrid_total_phase, self.threadsperblock,\
                                     self.CT, dt, self.c, current_time_step, \
                                     processes['DT'], processes['VT'], self.drift_order)
         
        if current_time_step%100==0:
            # check is the obtained self.fp is legal.
            whether_legal, illegal_type, value, which_species = check_legacy_of_distributions(self.f, self.particle_type, self.hbar, self.dt, self.degeneracy)

            if whether_legal == False:
                raise AssertionError("The algorithm fails to give correct distribution since species {} has got a wrong value {} = {}  . Try to decrease dt, or choose a proper and smooth initial distribution.".format(which_species, illegal_type, value)) 
                return 1
        return 0
    
    def PF_PT(self, processes, BEx, BEy, BEz, BBx, BBy, BBz, dt, current_time_step, total_time_steps):
        """Give the partial_f\partial_t.
        Params
        ======
        processes:
            whether to consider the Vlasov, Drift or Collision terms
        BEx, BEy, BEz, BBx, BBy, BBz:
            background EM fields, will be added to the medium generated EM fields
        dt:
            dt at t current levet
        current_time_step, total_time_steps:
            the current time step and the total time steps
        """
            
        self.current_time_step = current_time_step
        
#         if 1000 > self.current_time_step:
#             print('self.f.max(), self.charges, self.dx, self.dy, self.dz, self.electric_rho.min, max')
#             print(self.f.max(), self.charges, self.dx, self.dy, self.dz)
            
        # calculate the external forces
        if processes['VT']>0.5:
            BEx, BEy, BEz, BBx, BBy, BBz = cupy.asarray(BEx), cupy.asarray(BEy), cupy.asarray(BEz),\
                                           cupy.asarray(BBx), cupy.asarray(BBy), cupy.asarray(BBz)
            self.external_forces(BEx, BEy, BEz, BBx, BBy, BBz)
            self.rho_J_to_EB()
        else:
            self.Fx, self.Fy, self.Fz = cupy.zeros_like(self.f), cupy.zeros_like(self.f), cupy.zeros_like(self.f)
            
        # if 'CT' is instantiated, calculate the collision terms at all levels
        if processes['CT'] > 0.5:
            # calculate collision term
            self.CT = Collision_term(self.flavor, self.collision_type, \
                                     self.particle_type, self.degeneracy, self.num_samples,\
                                     self.f, self.masses, self.total_phase_grids, self.rng_states,\
                                     self.num_particle_species, \
                                     self.npx, self.npy, self.npz, self.nx, self.ny, self.nz, self.dp, \
                                     self.half_px, self.half_py, self.half_pz, \
                                     self.dpx, self.dpy, self.dpz, \
                                     self.blockspergrid_total_phase, self.threadsperblock, \
                                     self.middle_npx, self.middle_npy, self.middle_npz, \
                                     self.number_momentum_levels,\
                                     self.hbar,self.c,self.lambdax,self.allowed_collision_type)
        else:
            # otherwise set collision to be zero
            self.CT = cupy.zeros_like(self.f)
     
        # evaluate Drift and Vlasov terms
        whether_terminate = self.DT_VT(self.f, dt, current_time_step, processes)
        return whether_terminate

    def take_boundary(self, bound_index):
        """Take the spatial ghost boundaries.
        There are 6 surfaces. 
        bound_index:
            the surface that needs to be taken.
        """
        
        # take the relavant boundaries as indicated
        return cupy.asnumpy(self.f[:,:,\
                             self.surface_index_taken[bound_index,0]:self.surface_index_taken[bound_index,1],\
                             self.surface_index_taken[bound_index,2]:self.surface_index_taken[bound_index,3],\
                             self.surface_index_taken[bound_index,4]:self.surface_index_taken[bound_index,5]])
    
    def reset_boundary(self, surface, bound_index):
        """Reset the spatial ghost boundary values.
        There are 6 surfaces, 12 edges and 8 corners.
        These boundaries must be reset according to the order: surface, edge and corner.
        Params
        ======
        surface:
            the surface that needs to be exchanged, should be consistent with the input bound_indicator.
        bound_index:
            the surface that needs to be replaced.
        """

        # replace the boundaries
        self.f[:,:,\
               self.surface_index[bound_index,0]:self.surface_index[bound_index,1],\
               self.surface_index[bound_index,2]:self.surface_index[bound_index,3],\
               self.surface_index[bound_index,4]:self.surface_index[bound_index,5]] = cupy.asarray(surface)
                
    def return_self(self, process):
        '''Return the parameters according to the given name.
        Params
        ======
        process:
            string, includes: 'Vlasov term','Drift term', 'Collision term', 'Distribution',
                              'Forces', "Electric rho/J", 
                              "EM fields on current region",
                              "EM fields to other regions", "number_rho/J"
        return
        ======
        the parameter indicated by process.
        Note that these data might be copied to GPU:0.'''

        if 'Collision term' == process:
            return self.CT.get()

        if 'Distribution' == process:
            return self.f.get()
        
        if 'Forces' == process:
            return self.Fx.get(), self.Fy.get(), self.Fz.get()
        
        if "Electric rho/J" == process:
            return self.electric_rho.get(), self.electric_Jx.get(), self.electric_Jy.get(), self.electric_Jz.get()
        
        if "number_rho/J" == process:
            self.get_rho_J()
            return self.number_rho.get(), self.number_Jx.get(), self.number_Jy.get(), self.number_Jz.get()
        
        if "EM fields on current region" == process:
            return self.Ex_total.get(), self.Ey_total.get(), self.Ez_total.get(), self.Bx_total.get(), self.By_total.get(), self.Bz_total.get()
        
        if "EM fields to other regions" == process:
            return self.Ex_dis, self.Ey_dis, self.Ez_dis, self.Bx_dis, self.By_dis, self.Bz_dis
        if "stream_ticker" == process:
            return self.stream_ticker.get()

        
    def reshape_to_six_dim(self):        
        '''# reshape the distribution functions 
        # This is an inplace operation, changing the values of f will
        # change the value of self.f as well'''
        self.f = self.f.reshape([self.number_momentum_levels, self.num_particle_species, \
                                 self.nx, self.ny, self.nz, self.npx, self.npy, self.npz])
        
    def reshape_to_one_dim(self):        
        '''# reshape the distribution functions 
        # This is an inplace operation, changing the values will
        # change the value of self.f as well'''
        self.f = self.f.reshape([self.number_momentum_levels, self.num_particle_species, -1])
        
    def update_EM_fields(self, Ex, Ey, Ez, Bx, By, Bz, index):
        '''update the electromagnetci fields accroding to the given
        params
        ======
        Ex, Ey, Ez, Bx, By, Bz:
            EM fields of shape nx*ny*nz
        index:
            the index of which it is to be exchanged'''
        
        # the index corresponds to which region contributes to this EM field
        self.Ex[index], self.Ey[index], self.Ez[index], \
        self.Bx[index], self.By[index], self.Bz[index] = \
        cupy.asarray(Ex), cupy.asarray(Ey), cupy.asarray(Ez), cupy.asarray(Bx), cupy.asarray(By), cupy.asarray(Bz)

    def get_rho_J(self):
        '''get the number density rho and current density J'''
        
        # Obtain number \rho and J from the distribution f
        # number_rho/number_Jx/number_Jy/number_Jz: [nx*ny*nz]
        self.number_rho, self.number_Jx, self.number_Jy, self.number_Jz = \
        density(self.f, self.num_particle_species, self.total_spatial_grids, \
                self.masses, self.charges, \
                self.total_phase_grids, self.momentum_volume_element, self.npx, self.npy, self.npz, \
                self.nx, self.ny, self.nz, self.half_px, self.half_py, self.half_pz, \
                self.dx, self.dy, self.dz, self.dpx, self.dpy, self.dpz, self.number_momentum_levels,\
                self.blockspergrid_total_phase, self.threadsperblock, self.c)
  
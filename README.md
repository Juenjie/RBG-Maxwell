# RBG-Maxwell
RBG-Maxwell is a general framework that simulates the (relativistic) collisional plasma systems in a fully consistent way on GPU clusters. Given the proper initial distributions of the relevant particles, RBG-Maxwell is able to produce the subsequent states of the system. 


## 1. The RBG-Maxwell Module

The package is coded by Jun-Jie Zhang  and improved by   Ming-Yan Sun. The project will be consistently maintained by Ming-Yan Sun and Jun-Jie Zhang. 

For further help, please contact us at zjacob@mail.ustc.edu.cn Jun-Jie Zhang and sunmingyan0301@163.com Ming-Yan Sun 

This package is free you can redistribute it and/or modify it under the terms of the Apache License Version 2.0, January 2004. [Licenses][http://www.apache.org/licenses/] 

To cite our work, please use the following three items:



```python
@article{10436541,
 author={Sun, Ming-Yan and Xu, Peng and Du, Tai-Jiao and Hu, Jin-Ming and Li, Jin-Jun and Zhang, Jun-Jie},
  journal={IEEE Transactions on Plasma Science}, 
  title={Utilization of the RBG-Maxwell Framework for Collisionless Plasma at Atmospheric Scales}, 
  year={2024},
  volume={52},
  number={2},
  pages={576-581},
  keywords={Plasmas;Mathematical models;Graphics processing units;Maxwell equations;Ions;Distribution functions;Boltzmann equation;Collionless plasma;graphics processing unit (GPU) computing;kinetic equation;RBG-maxwell},
  doi={10.1109/TPS.2024.3361448}}
}

```



```python
@article{PhysRevD.102.074011,
  title = {Towards a full solution of the relativistic Boltzmann equation for quark-gluon matter on GPUs},
  author = {Zhang, Jun-Jie and Wu, Hong-Zhong and Pu, Shi and Qin, Guang-You and Wang, Qun},
  journal = {Phys. Rev. D},
  volume = {102},
  issue = {7},
  pages = {074011},
  numpages = {17},
  year = {2020},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevD.102.074011},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.102.074011}
}

```



```python
@article{ZHANG2022108328,
  title = {JefiGPU: Jefimenko equations on GPU},
  author = {Jun-Jie Zhang and Jian-Nan Chen and Guo-Liang Peng and Tai-Jiao Du and Hai-Yan Xie},         
  journal = {Computer Physics Communications},
  volume = {276},
  pages = {108328},
  year = {2022},
  issn = {0010-4655},
  doi = {https://doi.org/10.1016/j.cpc.2022.108328},
  url = {https://www.sciencedirect.com/science/article/pii/S0010465522000467},
}
```


**RBG-Maxwell** is written in **python**, so the user needs to ensure that the following packages are installed before running the program

```python
import numba   
	----numba is a JIT compiler that can compile python functions into machine code；
import math;    
	----math provides a number of mathematical functions for floating point numbers；
import cupy;   
	----cupy is an implementation of NumPy-compatible multidimensional arrays on CUDA;
import ray;     
	----Ray is a distributed execution framework;
import random;  
	----random is a standard library that generates random numbers；
import numpy;  
	----NumPy is the base package for scientific computing in Python；
import os;      
	----os is a module in the Python standard library for accessing operating system functions；
import sys;     
	----sys is a module to handle the python runtime environment。
```

To start up, create a conda environment and install RBG-Maxwell:
```
# create a new environment
$: conda create -n RBG-Maxwell

# install relavant package sequentially
$: conda install numba
$: pip install -U ray
$: conda install cupy matplotlib
$: conda install jupyter nobteook

$ git clone https://github.com/Juenjie/RBG-Maxwell
$ cd JefiPIC
$ jupyter notebook
```
**Note that the installation of Ray requires pip and compatible python versions! Usually this can be solved by using a lower version of Python**

## 2. The more details
If you are looking to delve deeper into the specifics of RBG-Maxwell, we recommend you visit our published webpage at "Juenjie.github.io". There, we have meticulously detailed the functionalities of each RBG-Maxwell module, complete with concrete code details. Additionally, we have provided a systematic demonstration of the various examples that we have made available.

## 3、Usage via an example

​	**The following codes demonstrate an example of how to use RBG-Maxwell.**

#### 3.1、 Set the initial conditions

- 1、First, we need to invoke the following package：

```python
iimport warnings
warnings.filterwarnings("ignore")

# specify the system
from RBG_Maxwell.Collision_database.select_system import which_system

plasma_system = 'Fusion_system'
which_system(plasma_system)

from RBG_Maxwell.Collision_database.Fusion_system.collision_type import collision_type_for_all_species
from RBG_Maxwell.Unit_conversion.main import determine_coefficient_for_unit_conversion, unit_conversion
import numpy as np
from RBG_Maxwell.Plasma.main import Plasma
```

- 2、Then, we need to specify the unit conversion factor：
  - Determine the conversion factors for the International System of Units (IS) and the Flexible System of Units (FS) by configuring the spatial grid, velocity, and charge parameters.

```python
dx = dy = 10**(-5)
dz = 1.
dx_volume = dx*dy*dz 
# velocity is roughly 10**(6) m/s
v = 5*10**6

# charge
Q = 1.6*10**(-19) 

# momentum is roughly 10**(-30)kg*10**7m/s
momentum = 10**(-23)
# the momentum grid is set to be 
# npy=100, npx=npz=1, half_px=half_pz=half_py~10**(-23)
# hence dpy~10**(-26), dpx and dpz have no effect 
dp = (10**(-25)*10**(-23)*10**(-23))**(1/3)
dp_volume = dp*dp*dp
# the total number of particles are 5*10**(-13)/(1.6*10**(-19))
# put these particles in 71 spatial grids in z direction
# in 201 spatial grids in y direction
# and 100 momentum grids
# in each phase grid we have dn = 21.89755448111554
# the average value of distribution is roughly 
dn = 0.2189755448111554
f = dn/(dp**3*dx*dy*dz)
df = f

n_max = 5*10**(-13)/(1.6*10**(-19))

n_average = 5*10**(-13)/(1.6*10**(-19))/(10000)


v_max = 1.5*10**6

E = 1000
B = 5.5*10**(-5)

# time scale
dt = 10**(-13)

# Now find the coefficient
hbar, c, lambdax, epsilon0 = determine_coefficient_for_unit_conversion(dt, dx, dx_volume, dp, dp_volume,n_max, n_average, v_max, E, B)
conversion_table = \
unit_conversion('SI_to_LHQCD', coef_J_to_E=lambdax, hbar=hbar, c=c, k=1., epsilon0=epsilon0)
conversion_table_reverse = \
unit_conversion('LHQCD_to_SI', coef_J_to_E=lambdax, hbar=hbar, c=c, k=1., epsilon0=epsilon0)
```

For a detailed procedure of unit conversion you can refer to [Conversion](http://Juenjie.github.io/jekyll/2022-07-10-Conversion.html).

- 3、Next, we need to initialize the plasma system：
  - The primary dataset comprises the parameters for time-space discretization, grid quantities, particle classification, and collision classification.

```python
dt, dx, dy, dz = 10**(-13)*conversion_table['second'], \
                 10**(-5)*conversion_table['meter'], \
                 10**(-5)*conversion_table['meter'], \
                 10**(-5)*conversion_table['meter']

# we have only one type of particle e-
num_particle_species = 1

# treat the electron as classical particles
particle_type = np.array([0])

# masses, charges and degenericies are
masses, charges, degeneracy = np.array([9.11*10**(-31)*conversion_table['kilogram']]), \
                              np.array([-1.6*10**(-19)*conversion_table['Coulomb']]),\
                              np.array([1.])

# momentum grids
npx, npy, npz = 1, 201, 1

# half_px, half_py, half_pz
# momentum range for x and z direction are not import in this case
half_px, half_py, half_pz = np.array([9.11*10**(-31)*5*10**6*conversion_table['momentum']]), \
                            np.array([9.11*10**(-31)*5*10**6*conversion_table['momentum']]),\
                            np.array([9.11*10**(-31)*5*10**6*conversion_table['momentum']])

dpx, dpy, dpz = 2*half_px/npx, 2*half_py/npy, 2*half_pz/npz

# load the collision matrix
flavor, collision_type, particle_order = collision_type_for_all_species()
expected_collision_type = ['2TO2']
```

The parameters related to collisions can be found in [Collision_database](https://Juenjie.github.io/jekyll/2022-07-04-Collision_database.html). The program describes in detail how to set up different colliding plasmas. We also set up the quark-gluon plasma system and the fusion system in this program.

- 4、Set parallel calculation parameters for the plasma system：	
  - Including the number of Monte Carlo particles, the number of regions, the number of GPUs in the regions, etc.

```python

# number of spatial grids
# the maximum spatial gird is limited by CUDA, it's about nx*ny*nz~30000 for each card
nx_o, ny_o, nz_o = [1], [251], [111]

# value of the left boundary
# this is the 
x_left_bound_o, y_left_bound_o, z_left_bound_o = [-0.5*dx],\
                                                 [-125.5*dy],\
                                                 [-55.5*dz]

# number samples gives the number of sample points in MC integration
num_samples = 100

# Only specify one spatial region
number_regions = 1

# each spatial should use the full GPU, this number can be fractional if many regions are chosen
# and only one GPU is available
num_gpus_for_each_region = 0.1


# since only one region is specified, this will be empty
sub_region_relations = {'indicator': [[]],\
                        'position': [[]]}

# if np.ones are used, the boundaries are absorbing boundaries
# if np.zeros are used, it is reflection boundary
# numbers in between is also allowed
boundary_configuration = {}
for i_reg in range(number_regions):
    bound_x = np.ones([ny_o[i_reg], nz_o[i_reg]])
    bound_y = np.ones([nz_o[i_reg], nx_o[i_reg]])
    bound_z = np.ones([nx_o[i_reg], ny_o[i_reg]])
    boundary_configuration[i_reg] = (bound_x, bound_y, bound_z)
```

For details, please refer to [Plasma](https://Juenjie.github.io/jekyll/2022-07-03-Plasma.html).

- 5、Set the distribution function and boundary conditions of the plasma system

```python
num_momentum_levels = 1

# iniital distribution function
f = {}
for i_reg in range(number_regions):
    f[i_reg] = np.zeros([num_momentum_levels, num_particle_species,\
                         nx_o[i_reg], ny_o[i_reg], nz_o[i_reg], npx, npy, npz])


# The initial velocity of the electrons is 1.87683*10**6 m/s, corresponds to the momentum value
# 9.11*10**(-31)*1.87683*10**6*conversion_table['momentum'] ~ 408.770512.
# The following code specifies the momentum grid index
dpy = 2*half_py/npy
a = 9.11*10**(-31)*1.87683*10**6*conversion_table['momentum']
ipy = [i for i in range(npy) if (-half_py+dpy*(i-0.5))<=a<=(-half_py+dpy*(i+1))][0]
dn_dv = 5*10**(-14)/(1.6*10**(-19))/(101*dx*dy*dz*dpx*dpy*dpz)

f[0][0, 0, 0,9,5:106,0,ipy,0] = dn_dv


# reshape the distribution function in different regions
for i_reg in range(number_regions):
    f[i_reg] = f[i_reg].reshape([num_momentum_levels, num_particle_species,\
                                 nx_o[i_reg]*ny_o[i_reg]*nz_o[i_reg]*npx*npy*npz])

'''
We add an external magnetic field of 10 T in the +y direction
'''
BBy = [10*conversion_table['Tesla']*np.ones(nx_o[0]*ny_o[0]*nz_o[0])]
BEx, BEy, BEz, BBx, BBz = [0],[0],[0],[0],[0]

plasma = Plasma(f, dt, \
                nx_o, ny_o, nz_o, dx, dy, dz, boundary_configuration, \
                x_left_bound_o, y_left_bound_o, z_left_bound_o, \
                npx, npy, npz, half_px, half_py, half_pz,\
                masses, charges, sub_region_relations,\
                flavor, collision_type, particle_type,\
                degeneracy, expected_collision_type,\
                num_gpus_for_each_region,\
                hbar, c, lambdax, epsilon0, \
                num_samples = 100, drift_order = 1,\
                rho_J_method="raw", GPU_ids_for_each_region = ["2"])
```

#### 3.2、System evolution and results output

- Set the time step and perform the plasma system evolution.

```python
n_step = 10001
number_rho = []
EM = []
charged_rho = []
dis = []
VT= []
DT = []
import time
start_time = time.time()
for i_time in range(n_step):  
    
    # if i_time%1000 == 0:
    #     dis.append(plasma.acquire_values("Distribution"))            
    plasma.proceed_one_step(i_time, n_step, processes = {'VT':1., 'DT':1., 'CT':0.},\
                            BEx = BEx, BEy = BEy, BEz = BEz, BBx = BBx, BBy = BBy, BBz = BBz)
    if i_time%1000 == 0:     
        print('Updating the {}-th time step'.format(i_time))
        number_rho.append(plasma.acquire_values("number_rho/J"))
        charged_rho.append(plasma.acquire_values("Electric rho/J"))
    EM.append(plasma.acquire_values('EM fields on current region'))
end_time = time.time()
```

- Using pictures to show the evolution of the system.

```python
# spatial distribution
# spatial distribution
import matplotlib.pyplot as plt
xi, yi = np.mgrid[1:252:1,1:112:1]
fig, axes = plt.subplots(ncols=5, nrows=2, figsize = (15,5))
for jj in range(2):
    for kk in range(5):
        axes[jj,kk].pcolormesh(xi, yi, number_rho[(jj*5+kk+1)][0][0].reshape([nx_o[0],ny_o[0],nz_o[0]])[0])
        # axes[jj,kk].contour(xi, yi, data[jj*5+kk].sum(axis=-1)[0,1])
```


## License
The package is coded by Jun-Jie Zhang and Ming-yan Sun.

This package is free you can redistribute it and/or modify it under the terms of the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

For further questions and technical issues, please contact us at

zjacob@mail.ustc.edu.cn (Jun-Jie Zhang 张俊杰)

**File Structure**
```
RBG-Maxwell
│   README.md 
│   unit test dispersion effect in magnetized plasma.ipynb
│   unit test electron system 2D (plane wave 1st order).ipynb
│   unit test electron system 2D (plane wave 2nd order).ipynb
│   unit test electron system 2D (point expansion).ipynb
│   unit test electron system 2D (smooth point expansion).ipynb
│   unit test electron system 2D-zero-initial-velocity (plane wave).ipynb
│   unit test electron system particle_diffusion.ipynb
│
└───RBG_Maxwell
    │   Collision_database
    │   Collision_term
    │   EMsolver
    │   External_forces
    │   Macro_quantities
    │   Plasma
    │   Plasma_methods
    │   Plasma_single_GPU
    │   Unit_conversion
    │   Vlasov_Drifit_terms
    │   slover.py
    │   __init__.py

    two stream instability
    │   groth rate 1st.ipynb
    │   groth rate 2nd.ipynb
    │   phase diagram 1st.ipynb
    │   phase diagram 2nd.ipynb
    │   unit test two steram instability -1st.ipynb.ipynb
    │   unit test two steram instability -2nd.ipynb.ipynb
    │   unit test two steram instability -2nd.ipynb.ipynb
    │   unit test two steram instability -2nd.ipynb.ipynb
    │   conservation and energy conversion 1st.ipynb
    │   conservation and energy conversion 2nd.ipynb
    │   __init__.py

```

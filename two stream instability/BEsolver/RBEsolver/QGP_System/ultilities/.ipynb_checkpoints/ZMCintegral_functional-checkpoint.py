#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
from numba import cuda
import numba as nb
import random
import os
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import ray
import itertools
import time

@cuda.jit(device=True)
def cross(x1,y1,z1,x2,y2,z2,component):
    return (y1*z2-z1*y2,z1*x2-x1*z2,x1*y2-y1*x2)[round(component)]
    
@cuda.jit(device=True)    
def theta(xp,yp,zp,RA,gamma):
    if RA**2-xp**2-yp**2-(gamma*zp)**2>0.:
        return 1.
    else:
        return 0.
@cuda.jit(device=True)
def thetap(xp,yp,zp,RA,gamma):
    if RA**2-xp**2-yp**2>0.:
        return 1.
    else:
        return 0.
@cuda.jit(device=True)
def rho(xp,yp,zp,gamma,rho0,R0,d,RA):
    rp=math.sqrt(xp**2+yp**2+(gamma*zp)**2)
    return gamma*rho0/(1+math.exp((rp-R0)/d))*theta(xp,yp,zp,RA,gamma)
    
@cuda.jit(device=True)   
def rhoPM(xp,yp,zp,gamma,rho0,R0,d,RA,sign,b):
    return rho(xp-sign*b/2,yp,zp,gamma,rho0,R0,d,RA)
@cuda.jit(device=True)
def thetapPM(xp,yp,zp,RA,gamma,sign,b):
    return thetap(xp-sign*b/2,yp,zp,RA,gamma)
    
@cuda.jit(device=True)
def RPMx(xp,yp,zp,x,y,z,sign,t,v,b):
    return (x-xp)
@cuda.jit(device=True)
def RPMy(xp,yp,zp,x,y,z,sign,t,v,b):
    return (y-yp)
@cuda.jit(device=True)
def RPMz(xp,yp,zp,x,y,z,sign,t,v,b):
    return (z-(zp+sign*v*t))

@cuda.jit(device=True)
def denominator(sign,t,x,y,z,xp,yp,zp,Y0):
    re = ((y-yp)**2+(x-xp)**2+(t*math.sinh(sign*Y0)-(z-zp)*math.cosh(Y0))**2)
    if math.sqrt(re)<10.**-5:
        return 0.
    return re**(-3/2)
    
# In[2]:

def gpu_nums():
    # return all detected available gpus, int
    detected_gpu_id = ray.cluster_resources()['GPU']
    return detected_gpu_id


# In[3]:

def MCkernel(domain, parameters,num_parameters,parameter_shape,parameter_off_set,
             i_batch, batch_size, total_size, my_func, num_points,variables):
    
    @cuda.jit(device=True)
    def unravel(num_parameters,parameter_shape,id_,index):
        temp_id = id_
        # loop through each parameter
        for i_parameter in range(num_parameters):
            index_ele= temp_id%parameter_shape[i_parameter]
            index[i_parameter] =index_ele
            temp_id = temp_id//parameter_shape[i_parameter]
            
    @cuda.jit
    def integration_kernel(MCresult, domain, parameters, domain_range, total_size, batch_size, 
                           i_batch, rng_states, num_points, parameter_shape,parameter_off_set):

        thread_id = cuda.grid(1)
        
        if thread_id < batch_size:
            parameter_id = thread_id + i_batch * batch_size

            if parameter_id < total_size:

                # local array to save current parameter grid value
                aa = cuda.local.array(shape=num_parameters, dtype=nb.int32)
                for i in range(num_parameters):
                    aa[i] = 0
                unravel(num_parameters,parameter_shape,parameter_id,aa)
                
                # turn aa into one-dimensional
                for i in range(num_parameters-1):
                    aa[i+1] = aa[i+1]+parameter_off_set[i]

                # feed in parameter values to bb
                bb = cuda.local.array(shape=num_parameters, dtype=nb.float32)
                for i in range(num_parameters):
                    bb[i] = parameters[aa[i]]

                for i_sample in range(num_points):
                    
                    x_tuple = cuda.local.array(shape=dim, dtype=nb.float64)
                    
                    for j_dim in range(dim):
                        x_tuple[j_dim] = xoroshiro128p_uniform_float64(rng_states, thread_id) * domain_range[j_dim] + domain[j_dim][0]

                    # feed in values to user defined function, 
                    # and add all points' corresponding results in one chunk
                    cuda.atomic.add(MCresult, thread_id, fun(x_tuple, bb))
                    
    exec(variables, globals())
    exec(my_func, globals())

    dim = len(domain) 

    domain_range = np.array([(domain[j_dim][1] - domain[j_dim][0]) for j_dim in range(dim)], dtype=np.float64)
    domain = np.array(domain, dtype=np.float64)

    # Configure the threads and blocks
    threadsperblock = 32
    blockspergrid = (batch_size + (threadsperblock - 1)) // threadsperblock

    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=random.sample(range(0,100000),1)[0])

    rng_states = cuda.to_device(rng_states)
    
    parameter_off_set = cuda.to_device(parameter_off_set)
    
    MCresult = cuda.device_array([batch_size])
    
    # set MCresult to be zero
    @cuda.jit
    def set_zero(MCresult, batch_size):

        thread_id = cuda.grid(1)
        
        if thread_id < batch_size:
            MCresult[thread_id] = 0.
    
    set_zero[blockspergrid, threadsperblock](MCresult,batch_size)

    # Start the kernel
    integration_kernel[blockspergrid, threadsperblock](MCresult, domain, parameters, domain_range, total_size, batch_size, i_batch, 
                                                       rng_states, num_points, parameter_shape,parameter_off_set)
    
    # volumn of the domain
    volumn = np.prod(domain_range)
    
    MCresult = MCresult.copy_to_host()
    
    trial_result = volumn * MCresult / num_points

    
    return trial_result


# In[5]:

@ray.remote(num_gpus=1)
class MCintegral_functional():
    def __init__(self, my_func = None, domain = None, parameters = None, num_points = 20000, batch_size = 100, variables = None):
        
        self.func = my_func
        
        if parameters == None:
            raise AssertionError("Please enter appropriate parameters")
        '''
        dim_parameters = len(parameters)
        print(dim_parameters)
        inner_para_shape = [len(parameters[i]) for i in range(dim_parameters)]
        print(inner_para_shape)
        total_parameters = np.zeros([int(np.prod(inner_para_shape)), dim_parameters])
        print(np.shape(total_parameters))
        for i in range(len(inner_para_shape)):
            
        '''
        
        # detect if domain is in right form
        if domain == None:
            raise AssertionError("Please enter a domain")
        for temp in domain:
            if len(temp) != 2:
                raise AssertionError("Domain is incorrect")
            if temp[1] < temp[0]:
                raise AssertionError("Domain [a,b] should satisfy b>a")
                
        self.domain = domain
        
        self.num_points = num_points
        
        self.total_size = int(np.prod([len(parameters[i]) for i in range(len(parameters))]))
        
        self.parameters = parameters
        
        self.variables = variables
        
        self.batch_size = int(batch_size)
        
        self.num_batches =  int((self.total_size + (self.batch_size - 1)) // self.batch_size)
        
    def evaluate(self):

        # result for accumulating data
        MCresult = []
        
        parameter = self.parameters
        
        num_parameters = len(parameter)
        parameter_shape = np.array([len(array) for array in parameter])
        
        # parameter off set
        # [2,5,6] --> [2,7]
        parameter_off_set = []
        for i_parameter in range(num_parameters-1):
            parameter_off_set.append(np.sum(parameter_shape[:i_parameter+1]))
        parameter_off_set = np.array(parameter_off_set)
        
        parameters = []
        for i in parameter:
            parameters+=i
        parameters = np.array(parameters)

        # allocate time (internet performance)
        start_allocate = time.time()
        
        # loop through all gpus
        for i_batch in range(self.num_batches):
            
            # distributed calculation
            trial_result = MCkernel(self.domain, parameters,num_parameters,parameter_shape, parameter_off_set,
                                           i_batch, self.batch_size, self.total_size, self.func, self.num_points,self.variables)
            
            MCresult.append(trial_result)
        
        MCresult = np.concatenate(MCresult)
        
        return MCresult


# In[ ]:





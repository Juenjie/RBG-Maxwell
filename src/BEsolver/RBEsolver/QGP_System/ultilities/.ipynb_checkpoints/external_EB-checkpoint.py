# this is for installation by pip install zmcintegral
from BEsolver.RBEsolver.QGP_System.ultilities.ZMCintegral_functional import MCintegral_functional
import math,ray
import numpy as np

# user defined function
BSpectator = """
import math
# define a device function that should be used by cuda kernel
@cuda.jit(device=True)
def fun(x, para):
    xp = x[0]
    yp = x[1]
    zp = x[2]
    tt = para[5]
    xx = para[4]
    yy = para[3]
    zz = para[2]
    component = para[1]
    sign = para[0]    
    return alphaEM/e*math.sinh(sign*Y0)*rhoPM(xp,yp,zp,gamma,rho0,R0,d,RA,sign,impact_parameter)\
    *(1-thetapPM(xp,yp,zp,RA,gamma,-sign,impact_parameter))\
    *cross(0.,0.,1.,RPMx(xp,yp,zp,xx,yy,zz,sign,tt,v,impact_parameter),\
                    RPMy(xp,yp,zp,xx,yy,zz,sign,tt,v,impact_parameter),\
                    RPMz(xp,yp,zp,xx,yy,zz,sign,tt,v,impact_parameter),component)\
    *denominator(sign,tt,xx,yy,zz,xp,yp,zp,Y0)
"""

# user defined function
ESpectator = """ 
import math
# define a device function that should be used by cuda kernel
@cuda.jit(device=True)
def fun(x, para):
    xp = x[0]
    yp = x[1]
    zp = x[2]
    tt = para[5]
    xx = para[4]
    yy = para[3]
    zz = para[2]
    component = para[1]
    sign = para[0]  
    return alphaEM/e*math.cosh(Y0)*rhoPM(xp,yp,zp,gamma,rho0,R0,d,RA,sign,impact_parameter)\
    *(1-thetapPM(xp,yp,zp,RA,gamma,-sign,impact_parameter))\
    *(RPMx(xp,yp,zp,xx,yy,zz,sign,tt,v,impact_parameter),\
      RPMy(xp,yp,zp,xx,yy,zz,sign,tt,v,impact_parameter),\
      RPMz(xp,yp,zp,xx,yy,zz,sign,tt,v,impact_parameter))[round(component)]\
    *denominator(sign,tt,xx,yy,zz,xp,yp,zp,Y0)
"""

def EM_field_Spectator(x_left_bound, y_left_bound, z_left_bound,\
                       px_left_bound, py_left_bound, pz_left_bound,\
                       x_grid_size, y_grid_size, z_grid_size,\
                       px_grid_size, py_grid_size, pz_grid_size,\
                       impact_parameter,\
                       dx, dy, dz,\
                       dpx, dpy, dpz,\
                       dt,n_step,sample_points,variables,t0,\
                       num_GPUs):
    
    exec(variables, globals())
    
    # variables for parameter scan, except for time t
    [vari1, vari2, vari3, vari4, vari5] =  [[-1,1],\
                                            [0,1,2],\
                                            [z_left_bound + (i+0.5)*dz for i in range(z_grid_size)],\
                                            [y_left_bound + (i+0.5)*dy for i in range(y_grid_size)],\
                                            [x_left_bound + (i+0.5)*dx for i in range(x_grid_size)]]

    # divide n_step into num_GPUs segments
    if num_GPUs > 1:
        if n_step%(num_GPUs) !=0:
            nsize = n_step//(num_GPUs) + 1
            n_time = [nsize for i in range(n_step//nsize)]
            if sum(n_time) < n_step:
                n_time.append(n_step - sum(n_time))
        else:
            nsize = n_step//(num_GPUs)
            n_time = [nsize for i in range(num_GPUs)]
    else:
        nsize = n_step
        n_time = [nsize]

    vari = [[vari1, vari2, vari3, vari4, vari5,[i*dt+t0+i_GPU*nsize*dt for i in range(n_time[i_GPU])]] for i_GPU in range(len(n_time))]

    # the parameter grid 
    batch_size = [n_time[i_GPU]*x_grid_size*y_grid_size*z_grid_size*3*2 for i_GPU in range(len(n_time))]

    EMC, BMC = [], []
    for i_GPU in range(len(n_time)):
        
        # call MCintegral_functional
        E = MCintegral_functional.remote(my_func = ESpectator, 
                                         domain = [[-RA-impact_parameter/2,RA+impact_parameter/2],[-RA,RA],[-RA/gamma,RA/gamma]], 
                                         parameters = vari[i_GPU], 
                                         num_points = sample_points, 
                                         batch_size = batch_size[i_GPU],
                                         variables = variables)
        EMC.append(E.evaluate.remote())

        B = MCintegral_functional.remote(my_func = BSpectator, 
                                         domain = [[-RA-impact_parameter/2,RA+impact_parameter/2],[-RA,RA],[-RA/gamma,RA/gamma]], 
                                         parameters = vari[i_GPU], 
                                         num_points = sample_points, 
                                         batch_size = batch_size[i_GPU],
                                         variables = variables)
        BMC.append(B.evaluate.remote())

    for i_GPU in range(len(n_time)):
        # obtaining the result
        Efield = np.concatenate([np.reshape(ray.get(EMC[i_GPU]),\
                                                         [n_time[i_GPU],x_grid_size,y_grid_size,z_grid_size,3,2])\
                                 for i_GPU in range(len(n_time))])
        Bfield = np.concatenate([np.reshape(ray.get(BMC[i_GPU]),\
                                                         [n_time[i_GPU],x_grid_size,y_grid_size,z_grid_size,3,2])\
                                 for i_GPU in range(len(n_time))])
    return Efield, Bfield
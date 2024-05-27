from numba import cuda
import math
import cupy

'''
cuda kernel to calculate the curl of magnetic field B.
'''
@cuda.jit
def curl_kernel(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, total_spatial_grids, nx, ny, nz,\
                rho, electric_Jx, electric_Jy, electric_Jz, c, epsilon0,\
                GEx, GEy, GEz, GBx, GBy, GBz, dt):
    # threads loop in one dimension
    i_grid = cuda.grid(1)
            
    if i_grid < total_spatial_grids:
        iz = i_grid%nz
        iz_rest = i_grid//nz
        iy = iz_rest%ny
        iy_rest = iz_rest//ny
        ix = iy_rest%nx
        
        # enforce periodical boundary conditions, ipxPlus --> ipx+1
        ixPlus = (ix+1)%nx
        iyPlus = (iy+1)%ny
        izPlus = (iz+1)%nz

        # -1%3 should be 2, but cuda yields 0, so we use
        ixMinus = (ix+nx-1)%nx
        iyMinus = (iy+ny-1)%ny
        izMinus = (iz+nz-1)%nz
        
        curl_Bx = (Bz[ix,iyPlus,iz] - Bz[ix,iyMinus,iz])/(2*dy) - (By[ix,iy,izPlus] - By[ix,iy,izMinus])/(2*dz)
        curl_By = (Bx[ix,iy,izPlus] - Bx[ix,iy,izMinus])/(2*dz) - (Bz[ixPlus,iy,iz] - Bz[ixMinus,iy,iz])/(2*dx)
        curl_Bz = (By[ixPlus,iy,iz] - By[ixMinus,iy,iz])/(2*dx) - (Bx[ix,iyPlus,iz] - Bx[ix,iyMinus,iz])/(2*dy)
        
        # electron fluid velocity
        evx = (curl_Bx*c**2*epsilon0 - electric_Jx[i_grid])/(unit_charge*rho[i_grid])
        evy = (curl_By*c**2*epsilon0 - electric_Jy[i_grid])/(unit_charge*rho[i_grid])
        evz = (curl_Bz*c**2*epsilon0 - electric_Jz[i_grid])/(unit_charge*rho[i_grid])
        
        # electric field = v cross B
        GEx[i_grid] = evy*Bz[ix,iy,iz] - evz*By[ix,iy,iz]
        GEy[i_grid] = evz*Bx[ix,iy,iz] - evx*Bz[ix,iy,iz]
        GEz[i_grid] = evx*By[ix,iy,iz] - evy*Bx[ix,iy,iz]
        
        # magnetic field = nabla cross E
        curl_Ex = (Ez[ix,iyPlus,iz] - Ez[ix,iyMinus,iz])/(2*dy) - (Ey[ix,iy,izPlus] - Ey[ix,iy,izMinus])/(2*dz)
        curl_Ey = (Ex[ix,iy,izPlus] - Ex[ix,iy,izMinus])/(2*dz) - (Ez[ixPlus,iy,iz] - Ez[ixMinus,iy,iz])/(2*dx)
        curl_Ez = (Ey[ixPlus,iy,iz] - Ey[ixMinus,iy,iz])/(2*dx) - (Ex[ix,iyPlus,iz] - Ex[ix,iyMinus,iz])/(2*dy)
        
        GBx[i_grid] = -curl_Ex + dt*Bx[ix, iy, iz]
        GBy[i_grid] = -curl_Ey + dt*By[ix, iy, iz]
        GBz[i_grid] = -curl_Ez + dt*Bz[ix, iy, iz]

def plasma_method(rho_J_method, electric_rho, electric_Jx, electric_Jy, electric_Jz,
                  Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, nx, ny, nz, 
                  total_spatial_grids, blockspergrid_spatial, threadsperblock, c, epsilon0):
    '''
    Plasma method for obtaining rho and J. Currently, we only support "raw" and "quasi_neutral".
    '''

    if rho_J_method == 'raw':
        return electric_rho, electric_Jx, electric_Jy, electric_Jz
    elif rho_J_method == 'quasi_neutral':
        # in quasi_neutral rho is always zero and j needs to be subtracted by curl_of_B/mu0
        curl_Bx = cupy.zeros([total_spatial_grids])  
        curl_By = cupy.zeros([total_spatial_grids]) 
        curl_Bz = cupy.zeros([total_spatial_grids]) 
        print('B:',curl_Bx.max(),curl_By.max(),curl_Bz.max())
        curl_kernel[blockspergrid_spatial, threadsperblock](Bx.reshape([nx, ny, nz]), By.reshape([nx, ny, nz]), Bz.reshape([nx, ny, nz]), dx, dy, dz, curl_Bx, curl_By, curl_Bz, total_spatial_grids, nx, ny, nz)
        print('B:',curl_Bx.max(),curl_By.max(),curl_Bz.max(),Bx.max(),By.max(),Bz.max())
        electric_Jx -= curl_Bx*c**2*epsilon0
        electric_Jy -= curl_By*c**2*epsilon0
        electric_Jz -= curl_Bz*c**2*epsilon0
        print('after JFe:',electric_Jx.max(),'Je:', (curl_Bx*c**2*epsilon0).max())
        return electric_rho, electric_Jx, electric_Jy, electric_Jz

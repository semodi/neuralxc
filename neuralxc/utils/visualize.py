import numpy as np
import matplotlib.pyplot as plt

def plot_density_cut(rho, rmax=0, plane=2, height = 0):
    """Take a quick look at the loaded data in a particular plane

        Parameters
        ----------
        rmin,rmax: (3) list; upper and lower cutoffs
        plane = {0: yz-plane, 1: xz-plane, 2: xy-plane}
    """

    grid = np.array(rho.shape)
    RHO = _plane_cut(rho, plane, height, grid, rmax=rmax)

    fig = plt.figure()
    CS = plt.imshow(
        RHO, cmap=plt.cm.jet, origin='lower')
    plt.colorbar()
    # plt.show()
    return fig

def _plane_cut(data,
              plane,
              height,
              grid,
              rmax=0,
              return_mesh=False):
    """return_mesh = False : returns a two dimensional cut through 3d data
                     True : instead of data, 2d mesh is returned

      Parameters:
      ----------
         data
         plane = {0: yz-plane, 1: xz-plane, 2:xy-plane}
         unitcell = 3x3 array size of the unitcell
         grid = 3x1 array size of grid
         rmax = lets you choose the max grid cutoff
                       rmax = 0 means the entire grid is used
         return_mesh = boolean; decides wether mesh or cut through data is returned
    """

    if rmax == 0:
        mid_grid = (grid / 2).astype(int)
        rmax = mid_grid

    rmin = [0,0,0]
    # resolve the periodic boundary conditions
    x_pbc = list(range(-rmax[0], -rmin[0])) + list(range(rmin[0], rmax[0]))
    y_pbc = list(range(-rmax[1], -rmin[1])) + list(range(rmin[1], rmax[1]))
    z_pbc = list(range(-rmax[2], -rmin[2])) + list(range(rmin[2], rmax[2]))
    height = (int)(np.round(height))

    pbc_grids = [x_pbc, y_pbc, z_pbc]
    pbc_grids.pop(plane)

    A, B = np.meshgrid(*pbc_grids)

    indeces = [A, B]
    indeces.insert(plane, height)
    if not return_mesh:
        return data[indeces[0], indeces[1], indeces[2]]
    else:
        return A, B

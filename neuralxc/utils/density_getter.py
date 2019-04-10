"""Utility functions for real-space grid properties
"""
import numpy as np
import struct
from abc import ABC, abstractmethod
from ..base import ABCRegistry

class DensityGetterRegistry(ABCRegistry):
    REGISTRY = {}
    
class BaseDensityGetter(metaclass= DensityGetterRegistry):
    _registry_name = 'base'
    def __init__(self):
        pass

    @abstractmethod
    def get_density(self, file_path):
        pass

class SiestaDensityGetter(BaseDensityGetter):

    _registry_name = 'siesta'
    def __init__(self, binary):
        self._binary = binary

    def get_density(self, file_path):
        if self._binary:
            return SiestaDensityGetter.get_density_bin(file_path)
        else:
            return SiestaDensityGetter.get_density(file_path)

    @staticmethod
    def get_density_bin(file_path):
        """ Same as get_data for binary (unformatted) files
        """
        #Warning: Only works for cubic cells!!!
        #TODO: Implement for arb. cells

        bin_file = open(file_path, mode = 'rb')

        unitcell = '<I9dI'
        grid = '<I4iI'

        unitcell = np.array(struct.unpack(unitcell,
            bin_file.read(struct.calcsize(unitcell))))[1:-1].reshape(3,3)

        grid = np.array(struct.unpack(grid,bin_file.read(struct.calcsize(grid))))[1:-1]
        if (grid[0] == grid[1] == grid[2]) and grid[3] == 1:
            a = grid[0]
        else:
            raise Exception('get_data_bin cannot handle non-cubic unitcells or spin')

        block = '<' + 'I{}fI'.format(a)*a*a
        content = np.array(struct.unpack(block,bin_file.read(struct.calcsize(block))))

        rho = content.reshape(a+2, a, a, order = 'F')[1:-1,:,:]
        return rho, unitcell, grid[:3]

    @staticmethod
    def get_density_formatted(file_path):
        """Import data from RHO file (or similar real-space grid files)
        Data is saved in global variables.

        Structure of RHO file:
        first three lines give the unit cell vectors
        fourth line the grid dimensions
        subsequent lines give density on grid

        Parameters
        -----------
            file_path: string
                path to RHO (or RHOXC) file from which density is read

        Returns
        --------
            Density
        """
        rhopath = file_path
        unitcell = np.zeros([3, 3])
        grid = np.zeros([4])

        with open(file_path, 'r') as rhofile:

            # unit cell (in Bohr)
            for i in range(0, 3):
                unitcell[i, :] = rhofile.readline().split()

            grid[:] = rhofile.readline().split()
            grid = grid.astype(int)
            n_el = grid[0] * grid[1] * grid[2] * grid[3]

            # initiatialize density with right shape
            rho = np.zeros(grid)

            for z in range(grid[2]):
                for y in range(grid[1]):
                    for x in range(grid[0]):
                        rho[x, y, z, 0] = rhofile.readline()

        # closed shell -> we don't care about spin.
        rho = rho[:, :, :, 0]
        grid = grid[:3]
        return rho, unitcell, grid


def density_getter_factory(application, *args,**kwargs):
    """
    Factory for various DensityGetters

    Parameters:
    ------------
    application : str
        Should specify which application created density file

    Returns:
    --------
    DensityGetter

    """

    registry = BaseDensityGetter.get_registry()
    # symmetrizer_dict = dict(casimir = CasimirSymmetrizer)

    if not application in registry:
        raise Exception('DensityGetter: {} not registered'.format(application))

    return registry[application](*args,**kwargs)

"""Utility functions for real-space grid properties
"""
import numpy as np
import pandas as pd
import struct
from abc import ABC, abstractmethod
from ..base import ABCRegistry
import re
try:
    from pyscf.scf.chkfile import load_scf
    from pyscf import dft
    pyscf_found = True
except ModuleNotFoundError:
    pyscf_found = False


def get_dm(mo_coeff, mo_occ):
    """ Get density matrix"""
    return np.einsum('ij,j,jk -> ik', mo_coeff, mo_occ, mo_coeff.T)


class DensityGetterRegistry(ABCRegistry):
    REGISTRY = {}


class BaseDensityGetter(metaclass=DensityGetterRegistry):
    _registry_name = 'base'

    def __init__(self):
        pass

    @abstractmethod
    def get_density(self, file_path):
        pass


class CubeDensityGetter(BaseDensityGetter):

    _registry_name = 'cube'

    def __init__(self, **kwargs):
        pass

    def get_density(self, file_path, return_dict=False):

        rho = pd.read_csv(file_path, delim_whitespace=True, skiprows=9, header=None)
        mask = (~rho.isna()).values.flatten()
        rho = rho.values.flatten()
        rho = rho[mask]
        grid_dh = pd.read_csv(file_path, delim_whitespace=True, skiprows=3, header=None, nrows=3).values
        grid = grid_dh[:, 0].astype(int)
        unitcell = grid_dh[:, 1:] * grid

        rho = rho.reshape(*grid)
        res = [rho, unitcell, grid]

        if return_dict:
            return {'rho': res[0], 'unitcell': res[1], 'grid': res[2]}
        else:
            return res


class CP2KDensityGetter(CubeDensityGetter):

    _registry_name = 'cp2k'


class PySCFDensityGetter(BaseDensityGetter):

    _registry_name = 'pyscf'

    def __init__(self, binary=None, valence=False, **kwargs):
        self.valence = valence

    def get_density(self, file_path, return_dict=False):
        mol, results = load_scf(file_path)
        res = get_dm(results['mo_coeff'], results['mo_occ']), mol, (results['mo_coeff'], results['mo_occ'])
        if self.valence:
            print('Using only valence density'.format(self.valence))
            core = 0
            for aidx, _ in enumerate(mol.atom):
                charge = mol.atom_charge(aidx)
                if charge > 10:
                    core += 2
                elif charge > 2:
                    core += 1
            results['mo_occ'][:core] = 0

        if return_dict:
            return {'rho': res[0], 'mol': res[1], 'mf': res[2]}
        else:
            return res


class PySCFRadDensityGetter(BaseDensityGetter):

    _registry_name = 'pyscf_rad'

    def __init__(self, binary=None, valence=False, **kwargs):
        self.valence = valence

    def get_density(self, file_path, return_dict=False):
        mol, results = load_scf(file_path)
        if self.valence:
            print('Using only valence density'.format(self.valence))
            core = 0
            for aidx, _ in enumerate(mol.atom):
                charge = mol.atom_charge(aidx)
                if charge > 10:
                    core += 2
                elif charge > 2:
                    core += 1
            results['mo_occ'][:core] = 0

        dm = get_dm(results['mo_coeff'], results['mo_occ'])
        mf = dft.RKS(mol)
        mf.xc = 'PBE'
        mf.grids.level = 4
        mf.grids.build()
        rho = dft.numint.get_rho(mf._numint, mol, dm, mf.grids)
        grid_coords = mf.grids.coords
        grid_weights = mf.grids.weights
        res = rho, grid_coords, grid_weights
        if return_dict:
            return {'rho': res[0], 'grid_coords': res[1], 'grid_weights': res[2]}
        else:
            return res


class SiestaDensityGetter(BaseDensityGetter):

    _registry_name = 'siesta'

    def __init__(self, binary, **kwargs):
        self._binary = binary

    def get_density(self, file_path, return_dict=False):
        if self._binary:
            res = SiestaDensityGetter.get_density_bin(file_path)
        else:
            res = SiestaDensityGetter.get_density(file_path)

        if return_dict:
            return {'rho': res[0], 'unitcell': res[1], 'grid': res[2]}
        else:
            return res

    @staticmethod
    def get_density_bin(file_path):
        """ Same as get_data for binary (unformatted) files
        """
        #Warning: Only works for cubic cells!!!
        #TODO: Implement for arb. cells

        bin_file = open(file_path, mode='rb')

        unitcell = '<I9dI'
        grid = '<I4iI'

        unitcell = np.array(struct.unpack(unitcell, bin_file.read(struct.calcsize(unitcell))))[1:-1].reshape(3, 3)

        grid = np.array(struct.unpack(grid, bin_file.read(struct.calcsize(grid))))[1:-1]
        if (grid[0] == grid[1] == grid[2]) and grid[3] == 1:
            a = grid[0]
        else:
            raise Exception('get_data_bin cannot handle non-cubic unitcells or spin')

        block = '<' + 'I{}fI'.format(a) * a * a
        content = np.array(struct.unpack(block, bin_file.read(struct.calcsize(block))))

        rho = content.reshape(a + 2, a, a, order='F')[1:-1, :, :]
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

    def get_forces(self, path, n_atoms=-1):
        """find forces in siesta .out file for first n_atoms atoms
        """
        with open(path, 'r') as infile:
            infile.seek(0)

            p = re.compile("siesta: Atomic forces \(eV/Ang\):\nsiesta:.*siesta:    Tot ", re.DOTALL)
            p2 = re.compile(" 1 .*siesta: -", re.DOTALL)
            alltext = p.findall(infile.read())
            alltext = p2.findall(alltext[0])
            alltext = alltext[0][:-len('\nsiesta: -')]
            forces = []
            for i, f in enumerate(alltext.split()):
                if i % 5 == 0: continue
                if f == 'siesta:': continue
                forces.append(float(f))
        forces = np.array(forces).reshape(-1, 3)
        if n_atoms == -1:
            return forces
        else:
            return forces[:n_atoms]


def density_getter_factory(application, *args, **kwargs):
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

    return registry[application](*args, **kwargs)

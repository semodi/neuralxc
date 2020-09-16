"""
neuralxc.py
Implementation of a machine learned density functional

Handles the primary interface that can be accessed by the electronic structure code
and all other relevant classes
"""

import numpy as np
from .ml.network import load_pipeline
from .ml.network import NetworkEstimator
from .projector import DensityProjector
from .symmetrizer import symmetrizer_factory
from .utils.visualize import plot_density_cut
from .constants import Rydberg, Bohr, Hartree
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import os
import time
import traceback
from periodictable import elements as element_dict
from .timer import timer
import neuralxc.config as config
from glob import glob
try:
    import torch
except ModuleNotFoundError:
    pass
agnostic_dict = {i: 'X' for i in np.arange(500)}


def prints_error(method):
    """ Decorator:forpy only prints stdout, no error messages,
    therefore print each error message to stdout instead
    """
    def wrapper_print_error(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as e:
            print(''.join(traceback.format_tb(e.__traceback__)))
            print('NeuralXC: ', e)
            raise (e)

    return wrapper_print_error


@prints_error
def get_nxc_adapter(kind, path, options={}):
    """ Adapter factory for NeuralXC
    """
    kind = kind.lower()
    adapter_dict = {'pyscf': PySCFNXC, 'pyscf_rad':PySCFRadNXC}
    if not kind in adapter_dict:
        raise ValueError('Selected Adapter not available')
    else:
        adapter = adapter_dict[kind](path, options)
    return adapter


class NXCAdapter(ABC):
    @prints_error
    def __init__(self, path, options={}):
        # from mpi4py import MPI
        path = ''.join(path.split())
        if path[-len('.jit'):] == '.jit' or path[-len('.jit/'):]== '.jit/' :
            self._adaptee = NeuralXCJIT(path)
        else:
            self._adaptee = NeuralXC(path)
        self.initialized = False

        # This complicated structure is necessary because of forpy, which
        # for some reason doesn't let us access the dict by strings
        workers = 1
        for key in options:
            if key == 'max_workers':
                workers = options[key]
        if workers > 1:
            timer.threaded = True
        self._adaptee.max_workers = int(workers)

        print('NeuralXC: Using {} thread(s)'.format(self._adaptee.max_workers))

    @abstractmethod
    def get_V(self):
        pass


class PySCFNXC(NXCAdapter):
    def initialize(self, mol):
        self.initialized = True
        self._adaptee.initialize(mol=mol)

    def get_V(self, dm):
        E, V = self._adaptee.get_V(rho=dm)
        E /= Hartree
        V /= Hartree
        return E, V

class PySCFRadNXC(NXCAdapter):

    def initialize(self, grid_coords, grid_weights, mol):
        self.initialized = True
        self.grid_weights = np.array(grid_weights)
        self._adaptee.initialize(unitcell=np.array(grid_coords), grid=np.array(grid_weights),
        positions=mol.atom_coords(), species=[mol.atom_symbol(i) for i in range(mol.natm)])

    def get_V(self, rho):
        E, V = self._adaptee.get_V(rho=rho)
        E /= Hartree
        V /= Hartree
        return E, V

class NeuralXC():

    def __init__(self, path):
        model_paths = glob(path + '/*')
        self.basis_models = {}
        self.projector_models = {}
        self.energy_models = {}
        self.spec_agn = False
        print('NeuralXC: Instantiate NeuralXC, using jit model')
        print('NeuralXC: Loading model from ' + path)

        for mp in model_paths:
            if 'basis' in os.path.basename(mp):
                self.basis_models[mp.split('_')[-1]] =\
                 torch.jit.load(mp)
            if 'projector' in  os.path.basename(mp):
                self.projector_models[mp.split('_')[-1]] =\
                 torch.jit.load(mp)
            if 'xc' in os.path.basename(mp):
                self.energy_models[mp.split('_')[-1]] =\
                 torch.jit.load(mp)
            if 'AGN' in os.path.basename(mp):
                self.spec_agn = True
        print('NeuralXC: Model successfully loaded')

    @prints_error
    def initialize(self, **kwargs):
        """Parameters
        ------------------
        unitcell, array float
        	Unitcell in bohr
        grid, array float
        	Grid points per unitcell
        positions, array float
        	atomic positions
        species, list string
        	atomic species (chem. symbols)
        """
        timer.start('MD step')
        self.projector_kwargs = kwargs
        periodic = (kwargs['unitcell'].shape == (3,3)) #TODO: this is just a workaround
        self.unitcell = torch.from_numpy(kwargs['unitcell']).double()
        # self.unitcell_inv = torch.inverse(self.unitcell).detach().numpy()
        self.epsilon = torch.zeros([3,3]).double()
        self.epsilon.requires_grad = True
        if periodic:
            self.unitcell_we = torch.mm((torch.eye(3) + self.epsilon), self.unitcell)
        else:
            self.unitcell_we = self.unitcell
        self.grid = torch.from_numpy(kwargs['grid']).double()
        self.positions = torch.from_numpy(kwargs['positions']).double()
        self.positions_we = torch.mm(torch.eye(3) + self.epsilon, self.positions.T).T
        # self.positions = torch.mm(self.positions_scaled,self.unitcell)
        self.species = kwargs['species']
        if self.spec_agn:
            self.species = ['X' for s in self.species]
        if periodic:
            U = torch.einsum('ij,i->ij', self.unitcell, 1/self.grid)
            self.V_cell = torch.abs(torch.det(U))
            self.V_ucell = torch.abs(torch.det(self.unitcell)).detach().numpy()
            self.my_box = torch.zeros([3,2])
            self.my_box[:,1] = self.grid
        else:
            self.V_cell = self.grid
            self.V_ucell = 1
            self.my_box = torch.zeros([3,2])
            self.my_box[:,1] = 1

        with torch.jit.optimized_execution(should_optimize=True):
            self.compute_basis(False)

    @prints_error
    def compute_basis(self, positions_grad=False):
        self.positions.requires_grad = positions_grad
        self.positions_we = torch.mm(torch.eye(3) + self.epsilon, self.positions.T).T
        if positions_grad:
            unitcell = self.unitcell_we
            positions = self.positions_we
        else:
            unitcell = self.unitcell
            positions = self.positions

        self.radials = []
        self.angulars = []
        self.boxes = []
        timer.start('build_basis')
        for pos, spec in zip(positions, self.species):
            rad, ang, box = self.basis_models[spec](pos, unitcell, self.grid, self.my_box)
            self.radials.append(rad)
            self.angulars.append(ang)
            self.boxes.append(box)
        timer.stop('build_basis')

    @prints_error
    def get_V(self, rho, calc_forces=False):

        if calc_forces:
            unitcell = self.unitcell_we
            positions = self.positions_we
        else:
            unitcell = self.unitcell
            positions = self.positions

        with torch.jit.optimized_execution(should_optimize=True):
            if calc_forces:
                timer.start('get_V_forces')
                self.compute_basis(True)
            else:
                timer.start('get_V')
            self.descriptors = {spec:[] for spec in self.species}
            rho = torch.from_numpy(rho).double()
            rho.requires_grad = True
            e_list = []
            for pos, spec, rad, ang, box in zip(positions, self.species,
                                                self.radials, self.angulars, self.boxes):
                e_list.append(self.energy_models[spec](
                    self.projector_models[spec](rho, pos,
                                                unitcell,
                                                self.grid,
                                                rad, ang, box).unsqueeze(0)
                                                )
                                            )

                # e_list[-1].backward()
            E = torch.sum(torch.cat(e_list))
            E.backward()
            V = (rho.grad/self.V_cell).detach().numpy()
            if calc_forces:
                V = [V, np.concatenate([-self.positions.grad.detach().numpy(),
                    self.epsilon.grad.detach().numpy()/self.V_ucell])]

                timer.stop('MD step')
                timer.stop('master')
                timer.stop('get_V_forces')
                timer.create_report('NXC_TIMING_JIT')
                timer.start('master')
            else:
                timer.stop('get_V')
            return E.detach().numpy(), V

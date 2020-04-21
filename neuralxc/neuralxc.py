"""
neuralxc.py
Implementation of a machine learned density functional

Handles the primary interface that can be accessed by the electronic structure code
and all other relevant classes
"""

import numpy as np
from .ml.network import load_pipeline
from .ml.network import NetworkEstimator
from .projector import DensityProjector, DeltaProjector
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
import torch
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
    adapter_dict = {'siesta': SiestaNXC, 'pyscf': PySCFNXC}
    if not kind in adapter_dict:
        raise ValueError('Selected Adapter not available')
    else:
        adapter = adapter_dict[kind](path, options)
    return adapter


@prints_error
def get_V(nxc, *args):
    """ Covenience function. Syntactically it might be easier (e.g. from
    Fortran) to function on module level than as a class member
    """
    res = nxc.get_V(*args)


class NXCAdapter(ABC):
    @prints_error
    def __init__(self, path, options={}):
        # from mpi4py import MPI
        path = ''.join(path.split())
        if path[-len('.jit'):] == '.jit' :
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


class SiestaNXC(NXCAdapter):
    @prints_error
    def initialize(self, rho, unitcell, grid, positions, elements):
        elements = np.array([str(element_dict[e]) for e in elements])
        unitcell = unitcell.T
        positions = positions.T
        if hasattr(self._adaptee,'_pipeline'):
            model_elements = [key for key in self._adaptee._pipeline.get_basis_instructions() if len(key) == 1]
            self.element_filter = np.array([(e in model_elements) for e in elements])
        else:
            self.element_filter = np.array([True]*len(positions))
        positions = positions[self.element_filter]
        elements = elements[self.element_filter]
        self._adaptee.initialize(unitcell=unitcell, grid=grid, positions=positions, species=elements)
        use_drho = False
        if hasattr(self._adaptee, '_pipeline'):
            if self._adaptee._pipeline.get_basis_instructions().get('extension', 'RHOXC') == 'DRHO':
                use_drho = True
                print('NeuralXC: Using DRHO')
                rho_reshaped = rho.reshape(*grid[::-1]).T
                self._adaptee.projector = DeltaProjector(self._adaptee.projector)
                self._adaptee.projector.set_constant_density(rho_reshaped, positions, elements)
            else:
                print('NeuralXC: Using RHOXC')
        else:
            print('NeuralXC: Using RHOXC')

        self.initialized = True

        return use_drho

    @prints_error
    def set_max_workers(self, max_workers):
        self._adaptee.max_workers = max_workers

    @prints_error
    def get_V(self, rho, unitcell, grid, positions, elements, V, calc_forces=False):
        """Parameters
        ------------------
        rho, array, float
        	Electron density in real space
        unitcell, array float
        	Unitcell in bohr
        grid, array float
        	Grid points per unitcell
        positions, array float
        	atomic positions
        elements, list of ints
        	atomic numbers
        calc_forces, bool
        	calculate force correction?

        Returns
        ---------
        E, (force_correction) np.ndarray
        	Machine learned potential

        Modifies
        --------
        V, shared memory with fortran code, modified in place

        Note
        -----
        Arrays should be provided in Fortran order
        """
        rho_reshaped = rho.reshape(*grid[::-1]).T
        if not self.initialized:
            raise Exception('Must call initialize before calling get_V')

        Enxc, Vnxc = self._adaptee.get_V(rho_reshaped, calc_forces=calc_forces)
        if calc_forces:
            self.force_correction = Vnxc[1][:-3].T / Rydberg
            self.stress_correction = Vnxc[1][-3:].T / Rydberg
            Vnxc = Vnxc[0]

        Enxc = Enxc / Rydberg
        Vnxc = Vnxc.real.T.reshape(-1, 1) / Rydberg
        V[:, :] = Vnxc + V
        print('NeuralXC: Enxc = {} eV'.format(Enxc * Rydberg))
        return Enxc

    @prints_error
    def correct_forces(self, forces):
        if hasattr(self, 'force_correction'):
            forces[:, self.element_filter] = forces[:, self.element_filter] + self.force_correction
        else:
            raise Exception('get_V with calc_forces = True has to be called before forces can be corrected')

    @prints_error
    def correct_stress(self, stress):
        if hasattr(self, 'stress_correction'):
            stress = stress + self.stress_correction
        else:
            raise Exception('get_V with calc_forces = True has to be called before stress can be corrected')


class NeuralXC():
    @prints_error
    def __init__(self, path=None, pipeline=None):
        global element_dict
        print('NeuralXC: Instantiate NeuralXC')
        if isinstance(path, str):
            if os.path.isfile(path):
                # If path contains the pipeline.pckl, ignore it
                path = os.path.dirname(path)
            print('NeuralXC: Load pipeline from ' + path)
            self._pipeline = load_pipeline(path)
        elif not (pipeline is None):
            self._pipeline = pipeline
        else:
            raise Exception('Either provide path to pipeline or pipeline')

        symmetrize_dict = {'basis': self._pipeline.get_basis_instructions()}
        symmetrize_dict.update(self._pipeline.get_symmetrize_instructions())
        self.symmetrizer = symmetrizer_factory(symmetrize_dict)
        if symmetrize_dict['basis'].get('spec_agnostic', False):
            element_dict = agnostic_dict
        self.max_workers = config.DefaultNThreads
        print('NeuralXC: Pipeline successfully loaded')

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
        symmetrize_dict = {'basis': self._pipeline.get_basis_instructions()}
        symmetrize_dict.update(self._pipeline.get_symmetrize_instructions())
        self.symmetrizer = symmetrizer_factory(symmetrize_dict)
        if symmetrize_dict['basis'].get('spec_agnostic', False):
            element_dict = agnostic_dict
        self.projector_kwargs = kwargs
        self.projector = DensityProjector(basis_instructions=self._pipeline.get_basis_instructions(), **kwargs)

    def _get_v_thread(self, dEdC, rho, positions, species, calc_forces=False):
        # print(positions, species)
        V = self.projector.get_V(dEdC, positions=positions, species=species, calc_forces=calc_forces, rho=rho)
        return V

    def _get_e_thread(self, rho, positions, species):
        # print(positions, species)
        positions = positions.reshape(-1, 3)
        species = [species]
        C = self.projector.get_basis_rep(rho, positions=positions, species=species)
        D = self.symmetrizer.get_symmetrized(C)
        E = self._pipeline.predict(D)[0]
        dEdD = self._pipeline.get_gradient(D)
        dEdC = self.symmetrizer.get_gradient(dEdD, C)
        return E, dEdC

    @prints_error
    def get_V(self, rho, calc_forces=False):
        """Parameters
        ------------------
        rho, array, float
        	Electron density in real space
        unitcell, array float
        	Unitcell in bohr
        grid, array float
        	Grid points per unitcell
        positions, array float
        	atomic positions
        species, list string
        	atomic species (chem. symbols)
        calc_forces, bool
        	calculate force and stress correction?

        Returns
        ------------
        E, V, (force_correction) np.ndarray
        	Machine learned potential, if calc_forces = True, force/stress corrections
            are returned as a (n_atoms+3 , 3) array.

        """

        E = 0
        if calc_forces:
            timer.start('get_V_forces')
        else:
            timer.start('get_V')

        if self.max_workers == 1:
            timer.start('project')
            C = self.projector.get_basis_rep(rho, **self.projector_kwargs)
            timer.stop('project')
            timer.start('ml_pipeline')
            D = self.symmetrizer.get_symmetrized(C)
            E = self._pipeline.predict(D)[0]
            dEdC = self.symmetrizer.get_gradient(self._pipeline.get_gradient(D))
            timer.stop('ml_pipeline')
            timer.start('build_V')
            V = self.projector.get_V(dEdC, calc_forces=calc_forces, rho=rho, **self.projector_kwargs)
            timer.stop('build_V')
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_rep = {
                    executor.submit(self._get_e_thread, rho, position, spec): spec
                    for position, spec in zip(self.projector_kwargs['positions'], self.projector_kwargs['species'])
                }

                dEdC = []
                for i, future in enumerate(future_to_rep):
                    results = future.result()
                    E += results[0]
                    dEdC.append(results[1])

                if not calc_forces:
                    V = 0
                    dEdC = np.array(dEdC)
                    future_to_rep = {
                        executor.submit(self._get_v_thread, dedc, rho, position.reshape(-1, 3), [spec], calc_forces):
                        spec
                        for dedc, position, spec in zip(dEdC, self.projector_kwargs['positions'],
                                                        self.projector_kwargs['species'])
                    }
                    for i, future in enumerate(future_to_rep):
                        results = future.result()
                        V += results
                else:
                    dEdC_dict = {}
                    for entry in dEdC:
                        for spec in entry:
                            if not spec in dEdC_dict:
                                dEdC_dict[spec] = []
                            dEdC_dict[spec].append(entry[spec])
                    for spec in dEdC_dict:
                        dEdC_dict[spec] = np.concatenate(dEdC_dict[spec], axis=1)

                    V = self.projector.get_V(dEdC_dict, calc_forces=calc_forces, rho=rho, **self.projector_kwargs)

        if calc_forces:
            timer.stop('get_V_forces')
            timer.stop('MD step')
            timer.stop('master')
            timer.create_report('NXC_TIMING')
            timer.start('master')
        else:
            timer.stop('get_V')
        return E, V


class NeuralXCJIT:


    def __init__(self, path):
        model_paths = glob(path + '/*')
        self.basis_models = {}
        self.projector_models = {}
        self.energy_models = {}
        # print(model_paths)
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
        self.unitcell = torch.from_numpy(kwargs['unitcell']).double()
        self.grid = torch.from_numpy(kwargs['grid']).double()
        self.positions = torch.from_numpy(kwargs['positions']).double()
        self.a = torch.norm(self.unitcell, dim=1).double() / self.grid
        U = torch.einsum('ij,i->ij', self.unitcell, 1/self.grid)
        self.V_cell = torch.det(U)
        self.species = kwargs['species']
        with torch.jit.optimized_execution(should_optimize=True):
            self.compute_basis(False)

    @prints_error
    def compute_basis(self, positions_grad=False):
        self.positions.requires_grad = positions_grad
        self.radials = []
        self.angulars = []
        timer.start('build_basis')
        for pos, spec in zip(self.positions, self.species):
            rad, ang = self.basis_models[spec](pos, self.unitcell, self.grid, self.a)
            self.radials.append(rad)
            self.angulars.append(ang)
        timer.stop('build_basis')

    @prints_error
    def get_V(self, rho, calc_forces=False):

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
            for pos, spec, rad, ang in zip(self.positions, self.species,
                                                self.radials, self.angulars):
                e_list.append(self.energy_models[spec](
                    self.projector_models[spec](rho, pos,
                                                self.unitcell,
                                                self.grid,
                                                self.a, rad, ang).unsqueeze(0)
                                                )
                                            )

                e_list[-1].backward()
            E = torch.sum(torch.cat(e_list))
            V = (rho.grad/self.V_cell).detach().numpy()
            if calc_forces:
                V = [V, np.concatenate([-self.positions.grad.detach().numpy(),np.zeros([3,3])])]
                timer.stop('MD step')
                timer.stop('master')
                timer.stop('get_V_forces')
                timer.create_report('NXC_TIMING_JIT')
                timer.start('master')
            else:
                timer.stop('get_V')
            return E.detach().numpy(), V

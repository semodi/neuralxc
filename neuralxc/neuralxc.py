"""
neuralxc.py
Implementation of a machine learned density functional

Handles the primary interface that can be accessed by the electronic structure code
and all other relevant classes
"""

import numpy as np
from .ml.network import load_pipeline
from .projector import DensityProjector
from .symmetrizer import symmetrizer_factory
from .utils.visualize import plot_density_cut
from .constants import Rydberg
from abc import ABC, abstractmethod
# from mpi4py import MPI

def prints_error(method):
    """ Decorator:forpy only prints stdout, no error messages,
    therefore print each error message to stdout instead
    """
    def wrapper_print_error(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as e:
            print(e)
            raise(e)

    return wrapper_print_error

def verify_type(obj):
    print('Type of object is:')
    print(obj)
    print(hasattr(obj, 'get_V' ))

@prints_error
def get_nxc_adapter(kind, path):
    """ Adapter factory for NeuralXC
    """
    kind = kind.lower()
    adapter_dict = {'siesta': SiestaNXC}
    if not kind in adapter_dict:
        print('Selected Adapter not available')
        raise ValueError('Selected Adapter not available')
    else:
        adapter = adapter_dict[kind](path)
    return adapter

@prints_error
def get_V(nxc, *args):
    """ Covenience function. Syntactically it might be easier (e.g. from
    Fortran) to function on module level than as a class member
    """
    res = nxc.get_V(*args)


class NXCAdapter(ABC):

    @prints_error
    def __init__(self, path):
        # from mpi4py import MPI
        path = ''.join(path.split())
        self._adaptee = NeuralXC(path)

    @abstractmethod
    def get_V(self):
        pass

class SiestaNXC(NXCAdapter):

    element_dict = {8: 'O', 1: 'H', 6: 'C'}

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
        elements = [self.element_dict[e] for e in elements]
        unitcell = unitcell.T
        positions = positions.T
        rho_reshaped = rho.reshape(*grid).T
        np.save('rho.npy',rho_reshaped)
        Enxc, Vnxc = self._adaptee.get_V(rho_reshaped, unitcell, grid,
                            positions, elements, calc_forces = calc_forces)
        if calc_forces:
            self.force_correction = Vnxc[1].T/Rydberg
            Vnxc = Vnxc[0]

        Enxc = Enxc/Rydberg
        Vnxc = Vnxc.real.T.reshape(-1,1)/Rydberg
        # print('Not correcting V!')
        V[:, :] = Vnxc + V
        print('Enxc = {} eV'.format(Enxc*Rydberg))
        return Enxc
    @prints_error
    def correct_forces(self, forces):
        try:
            if hasattr(self, 'force_correction'):
                forces[:] = forces + self.force_correction
            else:
                raise Exception('get_V with calc_forces = True has to be called before forces can be corrected')
        except Exception as e:
            print(e)
            raise(e)

class NeuralXC():

    def __init__(self, path = None, pipeline = None):

        print('Instantiate NeuralXC')
        try:
            if isinstance(path, str):
                print('Load pipeline from ' + path)
                self._pipeline = load_pipeline(path)
            elif not (pipeline is None):
                self._pipeline = pipeline
            else:
                raise Exception('Either provide path to pipeline or pipeline')
        except Exception as e:
            print(e)
            raise e
        print('Pipeline successfully loaded')

    def get_V(self, rho, unitcell, grid, positions, species, calc_forces= False):
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
        	calculate force correction?

        Returns
        ------------
        E, V, (force_correction) np.ndarray
        	Machine learned potential
        """
        projector = DensityProjector(unitcell, grid,
            self._pipeline.get_basis_instructions())

        symmetrize_dict = {'basis': self._pipeline.get_basis_instructions()}
        symmetrize_dict.update(self._pipeline.get_symmetrize_instructions())

        symmetrizer = symmetrizer_factory(symmetrize_dict)

        C = projector.get_basis_rep(rho, positions, species)

        D = symmetrizer.get_symmetrized(C)
        dEdC = symmetrizer.get_gradient(self._pipeline.get_gradient(D))
        E = self._pipeline.predict(D)[0]
        V = projector.get_V(dEdC, positions, species, calc_forces, rho)
        return E, V


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass

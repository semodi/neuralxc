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

# To test siesta integration:
def get_V(rho):
    result = np.linalg.norm(rho)
    return result

class NeuralXC():

    def __init__(self, path):
        self._pipeline = load_pipeline(path)

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
        return projector.get_V(dEdC, positions, species)

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass

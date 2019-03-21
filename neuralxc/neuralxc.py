"""
neuralxc.py
Implementation of a machine learned density functional

Handles the primary interface that can be accessed by the electronic structure code
and all other relevant classes
"""

import numpy as np
# To test siesta integration:
def get_V(rho):
    result = np.linalg.norm(rho)
    return result

class NeuralXC():

    def __init__(self):
        pass

    def get_V(self, rho, unitcell, grid, positions, species, calc_forces):
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
        pass

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass

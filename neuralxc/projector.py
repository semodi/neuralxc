from abc import ABC, abstractmethod

class BaseProjector(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_basis_rep(self, rho, unitcell, grid, positions, species, basis_instructions):
        """Calculates the basis representation for a given real space density

        Parameters
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
        basis_instructions, dict
        	Instructions that defines basis

        Returns
        ------------
        c, dict of np.ndarrays
        	Basis representation. One dict entry per atomic speciesk
        """
        pass

    @abstractmethod
    def get_V(self, dEdC, unitcell, grid, positions, species, basis_instructions, forces):
        """Calculates the basis representation for a given real space density

        Parameters
        ------------------
        dEdc , dict of numpy.ndarray
        unitcell, array float
        	Unitcell in bohr
        grid, array float
        	Grid points per unitcell
        positions, array float
        	atomic positions
        species, list string
        	atomic species (chem. symbols)
        basis_instructions, dict
        	Instructions that defines basis
        forces, bool
        	Calc. and return force corrections
        Returns
        ------------
        V, (force_correction) np.ndarray
        """
        pass

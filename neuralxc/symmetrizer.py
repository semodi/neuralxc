from abc import ABC, abstractmethod

class BaseSymmetrizer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_symmetrized(self, c, symmetrize_instructions):
        """
        Returns a symmetrized version of the descriptors c (from DensityProjector)

        Parameters
        ----------------
        c , dict of numpy.ndarrays
        	Electronic descriptors
        symmetrize_instructions, dict
        	Instructions on how to perform 	   symmetrization

        Returns
        ------------
        d, dict of numpy.ndarrays
        	Symmetrized descriptors
        """
        pass

    @abstractmethod
    def get_gradient(self, dEdD, symmetrize_instructions):
        """Uses chain rule to obtain dE/dc from dE/dd (unsymmetrized from symmetrized)

        Parameters
        ------------------
        dEdD : dict of np.ndarrays
        	dE/dD
        symmetrize_instructions: dict

        Returns
        -------------
        dEdc: dict of np.ndarrays
        """
        pass

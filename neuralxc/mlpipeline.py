from abc import ABC, abstractmethod

class BasePipeline(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, D):
        """Predicts energy and gradient

        Parameters
        ------------------

        D, dict of numpy.ndarrays
        	symmetrized descriptors

        Returns
        ------------
        E, dE/dd: float, dict of np.ndarrays
        	Energy and its derivative with respect to descriptors
        """
        pass

    @property
    def basis_instructions(self):
        return self._basis_instructions

    @basis_instructions.setter
    def basis_instructions(self, instructions):
        self._basis_instructions = instructions

    @property
    def symmetrize_instructions(self):
        return self._basis_instructions

    @symmetrize_instructions.setter
    def symmetrize_instructions(self, instructions):
        self._basis_instructions = instructions

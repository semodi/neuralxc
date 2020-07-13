from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from ..doc_inherit import doc_inherit
from sympy.physics.quantum.cg import CG
from sympy import N
import numpy as np
from ..formatter import expand
from ..base import ABCRegistry
from .symmetrizer import BaseSymmetrizer
try:
    import torch
    TorchModule = torch.nn.Module
except ModuleNotFoundError:
    class TorchModule:
         def __init__(self):
             pass

def convert_torch_wrapper(func):

    def wrapped_func(X, *args, **kwargs):
        X = torch.from_numpy(X)
        Y = func(X, *args, **kwargs)
        return Y.detach().numpy()

    return wrapped_func

class CasimirSymmetrizerTorch(TorchModule, BaseSymmetrizer):

    _registry_name = 'casimir_torch'

    def __init__(self, *args, **kwargs):

        BaseSymmetrizer.__init__(self, *args, **kwargs)
        TorchModule.__init__(self)

    def forward(self, C):
        self._symmetrize_function = self._symmetrize_function_bare
        return BaseSymmetrizer.get_symmetrized(self, C)

    def get_symmetrized(self, C):
        self._symmetrize_function = convert_torch_wrapper(self._symmetrize_function_bare)
        return BaseSymmetrizer.get_symmetrized(self, C)


    @staticmethod
    def _symmetrize_function_bare(c, n_l, n, *args):
        """ Returns the casimir invariants of the tensors stored in c

        Parameters:
        -----------

        c: np.ndarray of floats/complex
            Stores the tensor elements in the order (n,l,m)

        n_l: int
            number of angular momenta (not equal to maximum ang. momentum!
                example: if only s-orbitals n_l would be 1)

        n: int
            number of radial functions

        Returns
        -------
        np.ndarray
            Casimir invariants
        """
        c_shape = c.size()

        c = c.view(-1, c_shape[-1])
        casimirs = []
        idx = 0

        for n_ in range(0, n):
            for l in range(n_l):
                casimirs.append(torch.norm(c[:, idx:idx + (2 * l + 1)], dim=1)**2)
                idx += 2 * l + 1
        casimirs = torch.stack(casimirs).T

        return casimirs.view(*c_shape[:-1], -1)

    _symmetrize_function = _symmetrize_function_bare

    @staticmethod
    def _gradient_function(dEdd, c, n_l, n):
        pass


class MixedCasimirSymmetrizer(TorchModule, BaseSymmetrizer):

    _registry_name = 'mixed_casimir_torch'


    def __init__(self, *args, **kwargs):
        BaseSymmetrizer.__init__(self, *args, **kwargs)
        TorchModule.__init__(self)

    def forward(self, C):
        self._symmetrize_function = self._symmetrize_function_bare
        return BaseSymmetrizer.get_symmetrized(self, C)

    def get_symmetrized(self, C):
        self._symmetrize_function = convert_torch_wrapper(self._symmetrize_function_bare)
        return BaseSymmetrizer.get_symmetrized(self, C)

    @staticmethod
    def _symmetrize_function_bare(c, n_l, n, *args):
        """ Returns the casimir invariants with mixed radial channels
        of the tensors stored in c

        Parameters:
        -----------

        c: np.ndarray of floats/complex
            Stores the tensor elements in the order (n,l,m)

        n_l: int
            number of angular momenta (not equal to maximum ang. momentum!
                example: if only s-orbitals n_l would be 1)

        n: int
            number of radial functions

        Returns
        -------
        np.ndarray
            Casimir invariants
        """
        c_shape = c.size()

        c = c.view(-1, c_shape[-1])
        c = c.view(len(c), n, -1)
        casimirs = []

        for n1 in range(0, n):
            for n2 in range(n1, n):
                idx = 0
                for l in range(n_l):
                    casimirs.append(torch.sum(c[:,n1,idx:idx+(2*l+1)]*\
                                           c[:,n2,idx:idx+(2*l+1)],
                                            dim = -1))
                    idx += 2 * l + 1

        casimirs = torch.stack(casimirs).T

        return casimirs.view(*c_shape[:-1], -1)

    @staticmethod
    def _gradient_function(dEdd, c, n_l, n):
       pass

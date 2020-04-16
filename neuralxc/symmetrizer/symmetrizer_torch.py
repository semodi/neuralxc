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
import torch

def convert_torch_wrapper(func):

    def wrapped_func(X, *args, **kwargs):
        X = torch.from_numpy(X)
        Y = func(X, *args, **kwargs)
        return Y.detach().numpy()

    return wrapped_func

class CasimirSymmetrizerTorch(torch.nn.Module, BaseSymmetrizer):

    _registry_name = 'casimir_torch'

    def __init__(self, *args, **kwargs):

        BaseSymmetrizer.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)

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
        """Implements chain rule to obtain dE/dC from dE/dD
        (unsymmetrized from symmetrized)

        Parameters
        ------------------
        dEdD : np.ndarray
        	dE/dD

        c: np.ndarray
            Unsymmetrized basis representation

        n_l: int
            number of angular momenta (not equal to maximum ang. momentum!
                example: if only s-orbitals n_l would be 1)

        n: int
            number of radial functions

        Returns
        -------------
        dEdC: dict of np.ndarrays
        """
        pass
        # dEdd_shape = dEdd.shape
        # dEdd = dEdd.reshape(-1, dEdd.shape[-1])
        # c = c.reshape(-1, c.shape[-1])
        # casimirs_mask = np.zeros_like(c)
        # idx = 0
        # cnt = 0
        # for n_ in range(0, n):
        #     for l in range(n_l):
        #         casimirs_mask[:, idx:idx + (2 * l + 1)] = dEdd[:, cnt:cnt + 1]
        #         idx += 2 * l + 1
        #         cnt += 1
        #
        # grad = 2 * c * casimirs_mask
        # return grad.reshape(*dEdd_shape[:-1], grad.shape[-1])

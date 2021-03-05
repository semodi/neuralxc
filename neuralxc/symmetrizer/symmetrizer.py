"""
symmetrizer.py

Implements different symmetrizers. Symmetrizers ensure that descriptors are
invariant with respect to global rotations.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from neuralxc.base import ABCRegistry
from neuralxc.formatter import expand

TorchModule = torch.nn.Module


def convert_torch_wrapper(func):
    def wrapped_func(X, *args, **kwargs):
        made_tensor = False
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            made_tensor = True
        Y = func(X, *args, **kwargs)
        if made_tensor:
            return Y.detach().numpy()
        else:
            return Y

    return wrapped_func


class SymmetrizerRegistry(ABCRegistry):
    REGISTRY = {}


def Symmetrizer(symmetrize_instructions):
    """ Symmetrizer
    Parameters
    ----------
    symmetrize_instructions: dict
        Attributes needed to symmetrize input (such as angular momentum etc.)
    """

    sym_ins = symmetrize_instructions
    registry = BaseSymmetrizer.get_registry()
    if not 'symmetrizer_type' in sym_ins:
        raise Exception('symmetrize_instructions must contain symmetrizer_type key')

    symtype = sym_ins['symmetrizer_type']

    print('Using symmetrizer ', symtype)
    if not symtype in registry:
        raise Exception('Symmetrizer: {} not registered'.format(symtype))

    return registry[symtype](sym_ins)


class BaseSymmetrizer(TorchModule, BaseEstimator, TransformerMixin, metaclass=SymmetrizerRegistry):

    _registry_name = 'base'

    def __init__(self, symmetrize_instructions):
        """ Symmetrizer
        Parameters
        ----------
        symmetrize_instructions: dict
            Attributes needed to symmetrize input (such as angular momentum etc.)
        """

        TorchModule.__init__(self)
        self._attrs = symmetrize_instructions
        self._cgs = 0

    def forward(self, C):
        return BaseSymmetrizer.get_symmetrized(self, C)

    @abstractmethod
    def _symmetrize_function(c, n_l, n, *args):
        pass

    def get_params(self, *args, **kwargs):
        return {'symmetrize_instructions': self._attrs}

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        # If used in ML-pipeline X might actually contain (X, y)
        if isinstance(X, dict):
            self._attrs.update({'basis': X['basis_instructions']})
            X = X['data']

        if isinstance(X, tuple):
            symmetrized = self.get_symmetrized(X[0])
            return symmetrized, X[1]
        else:
            return self.get_symmetrized(X)

    def get_symmetrized(self, C):
        """
        Returns a symmetrized version of the descriptors c (from DensityProjector)

        Parameters
        ----------------
        C , dict of numpy.ndarrays or list of dict of numpy.ndarrays
            Electronic descriptors

        Returns
        ------------
        D, dict of numpy.ndarrays
            Symmetrized descriptors
        """
        self.C = C
        basis = self._attrs['basis']
        results = []

        for idx, key, data in expand(C):
            results.append({})
            results[idx][key] = self._symmetrize_function(*data, basis[key]['l'], basis[key]['n'], self._cgs)

        if not isinstance(C, list):
            return results[0]
        else:
            return results


class TraceSymmetrizer(BaseSymmetrizer):
    """ Symmetrizes density projections with respect to global rotations.
    
    :_registry_name: 'trace'
    """
    _registry_name = 'trace'

    def __init__(self, *args, **kwargs):

        BaseSymmetrizer.__init__(self, *args, **kwargs)

    def get_symmetrized(self, C):
        self._symmetrize_function = convert_torch_wrapper(self._symmetrize_function)
        return BaseSymmetrizer.get_symmetrized(self, C)

    @staticmethod
    def _symmetrize_function(c, n_l, n, *args):
        """ Returns the symmetrized version of c

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
        traces = []
        idx = 0

        for n_ in range(0, n):
            for l in range(n_l):
                traces.append(torch.norm(c[:, idx:idx + (2 * l + 1)], dim=1)**2)
                idx += 2 * l + 1
        traces = torch.stack(traces).T

        return traces.view(*c_shape[:-1], -1)


class MixedTraceSymmetrizer(BaseSymmetrizer):
    """
    :_registry_name: 'mixed_trace'
    """

    _registry_name = 'mixed_trace'

    def __init__(self, *args, **kwargs):
        BaseSymmetrizer.__init__(self, *args, **kwargs)

    def get_symmetrized(self, C):
        self._symmetrize_function = convert_torch_wrapper(self._symmetrize_function)
        return BaseSymmetrizer.get_symmetrized(self, C)

    @staticmethod
    def _symmetrize_function(c, n_l, n, *args):
        """ Return trace of c_m c_m' with mixed radial channels
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
        traces = []

        for n1 in range(0, n):
            for n2 in range(n1, n):
                idx = 0
                for l in range(n_l):
                    traces.append(torch.sum(c[:,n1,idx:idx+(2*l+1)]*\
                                           c[:,n2,idx:idx+(2*l+1)],
                                            dim = -1))
                    idx += 2 * l + 1

        traces = torch.stack(traces).T

        return traces.view(*c_shape[:-1], -1)


def symmetrizer_factory(symmetrize_instructions):
    """
    Factory for various Symmetrizers (Casimir, Bispectrum etc.).

    Parameters:
    ------------
    symmetrize_instructions : dict
        Should specify 'symmetrizer_type' ('trace','mixed_trace') and
        basis set information (angular momentum, no. radial basis functions)

    Returns:
    --------
    Symmetrizer

    """
    return Symmetrizer(symmetrize_instructions)

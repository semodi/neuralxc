from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from ..doc_inherit import doc_inherit
from sympy.physics.quantum.cg import CG
from sympy import N
import numpy as np
from ..formatter import expand
from ..base import ABCRegistry


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


class BaseSymmetrizer(BaseEstimator, TransformerMixin, metaclass=SymmetrizerRegistry):

    _registry_name = 'base'

    def __init__(self, symmetrize_instructions):
        """ Symmetrizer
        Parameters
        ----------
        symmetrize_instructions: dict
            Attributes needed to symmetrize input (such as angular momentum etc.)
        """

        self._attrs = symmetrize_instructions
        self._cgs = 0

    @abstractmethod
    def _symmetrize_function(c, n_l, n, *args):
        pass

    @abstractmethod
    def _gradient_function(dEdd, c, n_l, n):
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

    def get_gradient(self, dEdD, C=None):
        """Uses chain rule to obtain dE/dC from dE/dD (unsymmetrized from symmetrized)

        Parameters
        ------------------
        dEdD : dict of np.ndarrays or list of dict of np.ndarrays
        	dE/dD

        C : dict of np.ndarrays or list of dict of np.ndarrays
        	C

        Returns
        -------------
        dEdC: dict of np.ndarrays
        """
        if C == None:
            C = self.C

        basis = self._attrs['basis']

        results = [{}] * len(C)

        for idx, key, data in expand(dEdD, C):
            results[idx][key] = self._gradient_function(*data, basis[key]['l'], basis[key]['n'])
        if not isinstance(C, list):
            self.C = None
            return results[0]
        else:
            self.C = None
            return results


class CasimirSymmetrizer(BaseSymmetrizer):

    _registry_name = 'casimir'

    @staticmethod
    def _symmetrize_function(c, n_l, n, *args):
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

        c_shape = c.shape

        c = c.reshape(-1, c_shape[-1])
        casimirs = []
        idx = 0

        for n_ in range(0, n):
            for l in range(n_l):
                casimirs.append(np.linalg.norm(c[:, idx:idx + (2 * l + 1)], axis=1)**2)
                idx += 2 * l + 1
        casimirs = np.array(casimirs).T

        return casimirs.reshape(*c_shape[:-1], -1)

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
        dEdd_shape = dEdd.shape
        dEdd = dEdd.reshape(-1, dEdd.shape[-1])
        c = c.reshape(-1, c.shape[-1])
        casimirs_mask = np.zeros_like(c)
        idx = 0
        cnt = 0
        for n_ in range(0, n):
            for l in range(n_l):
                casimirs_mask[:, idx:idx + (2 * l + 1)] = dEdd[:, cnt:cnt + 1]
                idx += 2 * l + 1
                cnt += 1

        grad = 2 * c * casimirs_mask
        return grad.reshape(*dEdd_shape[:-1], grad.shape[-1])

class TotalCasimirSymmetrizer(CasimirSymmetrizer):

    _registry_name = 'total_casimir'
    _unit_test = False 

    @staticmethod
    def _symmetrize_function(c, n_l, n, *args):
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

        c_shape = c.shape

        c = c.reshape(-1, c_shape[-1])
        casimirs = []
        idx = 0

        for n_ in range(0, n):
            for l in range(n_l):
                casimirs.append(np.linalg.norm(c[:, idx:idx + (2 * l + 1)], axis=1)**2)
                idx += 2 * l + 1
        casimirs = np.array(casimirs).T

        return np.sum(casimirs.reshape(*c_shape[:-1], -1), axis=-1, keepdims=True)

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

class MixedCasimirSymmetrizer(BaseSymmetrizer):

    _registry_name = 'mixed_casimir'


    @staticmethod
    def _symmetrize_function(c, n_l, n, *args):
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
        c_shape = c.shape

        c = c.reshape(-1, c_shape[-1])
        c = c.reshape(len(c), n, -1)
        casimirs = []

        for n1 in range(0, n):
            for n2 in range(n1, n):
                idx = 0
                for l in range(n_l):
                    casimirs.append(np.sum(c[:,n1,idx:idx+(2*l+1)]*\
                                           np.conj(c[:,n2,idx:idx+(2*l+1)]),
                                            axis = -1).real)
                    idx += 2 * l + 1

        casimirs = np.array(casimirs).T

        return casimirs.reshape(*c_shape[:-1], -1)

    @staticmethod
    def _gradient_function(dEdd, c, n_l, n):
       pass
#
# class BispectrumSymmetrizer(Symmetrizer):
#
#     _registry_name = 'bispectrum'
#     def __init__(self, attrs):
#         print('WARNING! This class has not been thoroughly tested yet')
#         # super().__init__(attrs)
#         self._attrs = attrs
#         # Create array with Clebsch-Gordon coefficients
#         basis = attrs['basis']
#
#         n_l_max = 0
#         for spec in basis:
#             n_l_max = max(n_l_max, basis[spec]['l'])
#
#         self._cgs = cg_matrix(n_l_max)
#
#     @staticmethod
#     def _symmetrize_function(c, n_l, n, cgs=None):
#         """ Returns the bispectrum of the tensors stored in c
#
#         Parameters:
#         -----------
#
#         c: np.ndarray of floats/complex
#             Stores the tensor elements in the order (n,l,m)
#
#         n_l: int
#             number of angular momenta (not equal to maximum ang. momentum!
#                 example: if only s-orbitals n_l would be 1)
#
#         n: int
#             number of radial functions
#
#         cgs: np.ndarray, optional
#             Clebsch-Gordan coefficients, if not provided, calculated on-the-fly
#
#         Returns
#         -------
#         np.ndarray
#             Bispectrum
#         """
#         casimirs = CasimirSymmetrizer._symmetrize_function(c,n_l,n)
#
#         c_shape = c.shape
#
#         c = c.reshape(-1,c_shape[-1])
#         c = c.reshape(len(c),n,-1)
#         bispectrum = []
#         idx = 0
#
#         start = {}
#         for l in range(0, n_l):
#             start[l] = idx
#             idx += 2*l + 1
#
#         if not isinstance(cgs, np.ndarray):
#             cgs = cg_matrix(n_l)
#
#         for n in range(0, n):
#             for l1 in range(n_l):
#                 for l2 in range(n_l):
#                     for l in range(abs(l2-l1),min(l1+l2+1, n_l)):
#                         b = 0
#                         if np.linalg.norm(cgs[l1,:,l2,:,l,:]) < 1e-15:
#                             continue
#
#                         for m in range(-l,l+1):
#                             for m1 in range(-l1,l1+1):
#                                 for m2 in range(-l2,l2+1):
#                                     b +=\
#                                      np.conj(c[:,n,start[l] + m + l])*\
#                                      c[:,n,start[l1] + m1 + l1]*\
#                                      c[:,n,start[l2] + m2 + l2]*\
#                                      cgs[l1,m1,l2,m2,l,m]
#                                      # cgs[l1,l2,l,m1,m2,m]
#                         if np.any(abs(b.imag) > 1e-3):
#                             raise Exception('Not real')
#                         bispectrum.append(b.real.round(5))
#
#         bispectrum = np.array(bispectrum).T
#
#         bispectrum =  bispectrum.reshape(*c_shape[:-1], -1)
#         bispectrum = np.concatenate([casimirs, bispectrum], axis = -1)
#         return bispectrum


def symmetrizer_factory(symmetrize_instructions):
    """
    Factory for various Symmetrizers (Casimir, Bispectrum etc.).

    Parameters:
    ------------
    symmetrize_instructions : dict
        Should specify 'symmetrizer_type' ('casimir','bispectrum') and
        basis set information (angular momentum, no. radial basis functions)

    Returns:
    --------
    Symmetrizer

    """
    return Symmetrizer(symmetrize_instructions)


def cg_matrix(n_l):
    """ Returns the Clebsch-Gordan coefficients for maximum angular momentum n_l-1
    """
    lmax = n_l - 1
    cgs = np.zeros([n_l, 2 * lmax + 1, n_l, 2 * lmax + 1, n_l, 2 * lmax + 1], dtype=complex)

    for l in range(n_l):
        for l1 in range(n_l):
            for l2 in range(n_l):
                for m in range(-n_l, n_l + 1):
                    for m1 in range(-n_l, n_l + 1):
                        for m2 in range(-n_l, n_l + 1):
                            # cgs[l1,l2,l,m1,m2,m] = N(CG(l1,l2,l,m1,m2,m).doit())
                            cgs[l1, m1, l2, m2, l, m] = N(CG(l1, m1, l2, m2, l, m).doit())
    return cgs


# def to_casimirs_mixn(c, n_l, n):
#     """ Returns the casimir invariants with mixed radial channels
#     of the tensors stored in c
#
#     Parameters:
#     -----------
#
#     c: np.ndarray of floats/complex
#         Stores the tensor elements in the order (n,l,m)
#
#     n_l: int
#         number of angular momenta (not equal to maximum ang. momentum!
#             example: if only s-orbitals n_l would be 1)
#
#     n: int
#         number of radial functions
#
#     Returns
#     -------
#     np.ndarray
#         Casimir invariants
#     """
#     c_shape = c.shape
#
#     c = c.reshape(-1,c_shape[-1])
#     c = c.reshape(len(c),n,-1)
#     casimirs = []
#
#     for n1 in range(0, n):
#         for n2 in range(n1,n):
#             idx = 0
#             for l in range(n_l):
#                 casimirs.append(np.sum(c[:,n1,idx:idx+(2*l+1)]*\
#                                        np.conj(c[:,n2,idx:idx+(2*l+1)]), axis = -1).real)
#                 idx += 2*l + 1
#
#     casimirs = np.array(casimirs).T
#
#     return casimirs.reshape(*c_shape[:-1], -1)

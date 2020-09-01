from abc import ABC, abstractmethod
import numpy as np
from functools import reduce
import time
import math
from ..base import ABCRegistry
from ..timer import timer
import neuralxc.config as config
from ..utils import geom
import scipy.linalg
from periodictable import elements as element_dict
import periodictable
import torch
from torch.nn import Module as TorchModule
torch.set_default_dtype(torch.float64)
from .projector import EuclideanProjector, RadialProjector

class OrthoProjector(EuclideanProjector):
    """ Implements orthonormal basis functions on a euclidean grid.
    Radial basis is based on polynomials.
    """
    _registry_name = 'ortho'
    _unit_test = True

    @classmethod
    def g(cls, r, basis, a):
        r_o = basis['r_o']
        N = math.sqrt(720*r_o**(11+2*a)*1/((2*a+11)*(2*a+10)*(2*a+9)*(2*a+8)*(2*a+7)*\
                                           (2*a+6)*(2*a+5)))
        return r.pow(2) * (r_o - r).pow(a + 2) / N


    @staticmethod
    def orthogonalize(func, r, basis, W):
        r_o = basis['r_o']
        rad = []
        for k in torch.arange(0, W.size()[0]):
            rad.append(func(r, basis, (k + 1).double()))

        result = torch.einsum('ij,j...->i...', W, torch.stack(rad))
        result[:, r > r_o] = 0
        return result

    @classmethod
    def radials(cls, r, basis, W):
        '''
        Get orthonormal radial basis functions

        Parameters
        -------

            r: Tensor (npoints)
                radius
            basis: float
                basis info (r_o etc.)
            W: Tensor (nrad, nrad)
                orthogonalization matrix

        Returns
        -------
            Tensor (nrad, npoints)
                stacked radial functions
        '''
        return cls.orthogonalize(cls.g, r, basis, W)

    @classmethod
    def get_W(cls, basis):
        '''
        Get matrix to orthonormalize radial basis functions
        '''
        return scipy.linalg.sqrtm(np.linalg.pinv(cls.S(basis)))

    @classmethod
    def S(cls, basis):
        '''
        Overlap matrix between radial basis functions
        '''
        r_o = basis['r_o']
        nmax = basis['n']
        S_matrix = torch.zeros([nmax, nmax])
        r_grid = torch.linspace(0, r_o, 1000)
        dr = r_grid[1] - r_grid[0]
        for i in range(nmax):
            for j in range(i, nmax):
                S_matrix[i, j] = torch.sum(cls.g(r_grid, basis, i + 1) *\
                    cls.g(r_grid, basis, j + 1) * r_grid**2) * dr
        for i in range(nmax):
            for j in range(i + 1, nmax):
                S_matrix[j, i] = S_matrix[i, j]
        return S_matrix


class OrthoRadialProjector(RadialProjector, OrthoProjector):
    """ Implements orthonormal basis functions on a radial grid.
    Radial basis is based on polynomials.
    """
    _registry_name = 'ortho_radial'

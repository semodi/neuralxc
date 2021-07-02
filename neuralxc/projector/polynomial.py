"""
polynomial.py
Implements density projection basis with radial functions based on polynomials.
"""

import math

import numpy as np
import scipy.linalg
import torch
from opt_einsum import contract

from neuralxc.projector import EuclideanProjector, RadialProjector

torch.set_default_dtype(torch.float64)


class OrthoProjectorMixin():
    """ Implements orthonormal basis functions
    """
    def forward_basis(self, positions, unitcell, grid, my_box):
        """Creates basis set (for projection) for a single atom, on grid points

        Parameters
        ----------
        positions, Tensor (1, 3) or (3)
        	atomic position
        unitcell, Tensor (3,3)
        	Unitcell in bohr
        grid, Tensor (3)
        	Grid points per unitcell
        my_box, Tensor (3: euclid. directions, 2: upper and lower limits)
            Limiting box local gridpoints. Relevant if global grid is decomposed
            with MPI or similar.

        Returns
        --------
        rad, ang, mesh
            Stacked radial and angular functions as well as meshgrid
        """

        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        box, mesh = self.box_around(positions, basis['r_o'], my_box)
        rad, ang = self.get_basis_on_mesh(box, basis, self.W[self.species])
        return rad, ang, mesh

    def project_onto(self, rho, rads, angs, n_l):
        rho = rho.squeeze()
        rho = rho * self.V_cell.squeeze()
        if rho.ndim < 3:
            coeff_array = contract('li,ni,...i -> ...nl', angs, rads, rho)
        else:
            coeff_array = contract('lmijk,nijk,...ijk -> ...nlm', angs, rads, rho)

        return coeff_array.view(-1)

    def get_basis_on_mesh(self, box, basis, W):

        n_l = basis['l']
        R, Theta, Phi = box['radial']

        #Build angular part of basis functions
        angs = []
        for l in range(n_l):
            angs += self.angulars_real(l, Theta, Phi)

        angs = torch.stack(angs)
        rads = self.radials(R, basis, W)

        return rads, angs

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

        result = contract('ij,j...->i...', W, torch.stack(rad))
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


class OrthoEuclideanProjector(EuclideanProjector, OrthoProjectorMixin):
    """
    :_registry_name: 'ortho'
    """
    _registry_name = 'ortho'


class OrthoRadialProjector(RadialProjector, OrthoProjectorMixin):
    """
    :_registry_name: 'ortho_radial'
    """
    _registry_name = 'ortho_radial'

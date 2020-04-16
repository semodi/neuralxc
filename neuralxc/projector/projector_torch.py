from abc import ABC, abstractmethod
import numpy as np
from scipy.special import sph_harm
import scipy.linalg
from sympy import N
from functools import reduce
import time
import math
from ..doc_inherit import doc_inherit
from spher_grad import grlylm
from ..base import ABCRegistry
from numba import jit
from ..timer import timer
import neuralxc.config as config
from ..utils import geom_torch as geom
from .projector import BaseProjector, OrthoProjector
from periodictable import elements as element_dict
import periodictable
import torch


class DefaultProjectorTorch(torch.nn.Module, BaseProjector) :

    _registry_name = 'default_torch'

    def __init__(self, unitcell, grid, basis_instructions, **kwargs):
        """
        Parameters
        ------------------
        unitcell, array float
        	Unitcell in bohr
        grid, array float
        	Grid points per unitcell
        basis_instructions, dict
        	Instructions that defines basis
        """
        torch.nn.Module.__init__(self)
        self.basis = basis_instructions

        # Initialize the matrix used to orthonormalize radial basis
        W = {}
        for species in basis_instructions:
            if len(species) < 3:
                W[species] = self.get_W(basis_instructions[species])

        a = np.linalg.norm(unitcell, axis=1) / grid[:3]
        self.unitcell = torch.from_numpy(unitcell)
        self.grid = torch.from_numpy(grid).double()
        self.a = torch.from_numpy(a).double()
        self.W = {w:torch.from_numpy(W[w]) for w in W}

    def get_basis_rep(self, rho, positions, species, **kwargs):
        """Calculates the basis representation for a given real space density

        Parameters
        ------------------
        rho, array, float
        	Electron density in real space
        positions, array float
        	atomic positions
        species, list string
        	atomic species (chem. symbols)

        Returns
        ------------
        c, dict of np.ndarrays
        	Basis representation, dict keys correspond to atomic species.
        """
        rho = torch.from_numpy(rho)
        positions = torch.from_numpy(positions)
        species = torch.Tensor([getattr(periodictable,s).number for s in species])
        C = self.forward(rho, positions, species, self.unitcell, self.grid, self.a)
        return {spec: C[spec].detach().numpy() for spec in C}


    def forward(self, rho, positions, species, unitcell, grid, a):

        U = torch.einsum('ij,i->ij', unitcell, (grid[:3])**(-1))
        # a = torch.sqrt(torch.einsum('ij->i',unitcell**2)) / grid
        # a = torch.diag(U)
        # a = torch.from_numpy(np.array([0.20996965, 0.20996965, 0.20996965]))
        self.grid = grid
        self.V_cell = torch.det(U)
        # self.V_cell = 1
        self.U = torch.transpose(U,0,1)
        # self.U = U.T
        self.U_inv = torch.inverse(U)
        self.a = a
        self.all_angs = {}
        basis_rep = {}
        species = [str(element_dict[int(s.detach().numpy())]) for s in species]

        for pos, spec in zip(positions, species):
            if not spec in basis_rep:
                basis_rep[spec] = []

            idx = '{}{}{}{}'.format(spec, pos[0], pos[1], pos[2])
            basis = self.basis[spec]
            box = self.box_around(pos, basis['r_o'])
            projection, angs = self.project(rho, box, basis, self.W[spec],
                angs=self.all_angs.get(idx, None))

            basis_rep[spec].append(projection)

        for spec in basis_rep:
            basis_rep[spec] = torch.cat(basis_rep[spec], dim=0)

        return basis_rep

    @staticmethod
    def angulars_real(l, theta, phi):
        """ Angular function/angs (uses physics convention for angles)

        Parameters
        ----------
        l: int
            angular momentum quantum number
        m: int
            angular momentum projection

        theta: float or np.ndarray
            longitudinal angle
        phi: float or np.ndarray
            azimuthal angle

        Returns
        -------
        float or np.ndarray
            Value of angular function at provided point(s)
        """
        res = []
        for m in range(-l,l+1):
            res.append(geom.SH(l,m,theta,phi))
        return torch.stack(res)

    def project(self, rho, box, basis, W=None, return_dict=False, angs=None):
        '''
        Project the real space density rho onto a set of basis functions

        Parameters
        ----------
            rho: np.ndarray
                electron charge density on grid
            box: dict
                 contains the mesh in spherical and euclidean coordinates,
                 can be obtained with get_box_around()
            n_rad: int
                 number of radial functions
            n_l: int
                 number of spherical harmonics
            r_o: float
                 outer radial cutoff in Angstrom
            W: np.ndarray
                 matrix used to orthonormalize radial basis functions

        Returns
        --------
            dict
                dictionary containing the coefficients
        '''

        n_rad = basis['n']
        n_l = basis['l']
        r_o = basis['r_o']
        R, Theta, Phi = box['radial']
        Xm, Ym, Zm = box['mesh']

        # Automatically detect whether entire charge density or only surrounding
        # box was provided
        #Build angular part of basis functions
        angs = []
        for l in range(n_l):
            angs.append([])
            ang_l = self.angulars_real(l, Theta, Phi)
            for m in range(-l, l + 1):
                angs[l].append(ang_l[l + m])
        #Build radial part of b.f.
        # if not isinstance(W, np.ndarray):
            # W = self.get_W(basis)  # Matrix to orthogonalize radial basis

        rads = self.radials(R, basis, W)


        srho = rho[Xm.long(), Ym.long(), Zm.long()]

        #zero_pad_angs (so that it can be converted to numpy array):
        zeropad = torch.zeros_like(R)
        angs_padded = []
        for l in range(n_l):
            angs_padded.append(torch.stack([zeropad] * (n_l - l) + angs[l] + [zeropad] * (n_l - l)))

        angs_padded = torch.stack(angs_padded)
        rads = rads * self.V_cell
        coeff_array = torch.einsum('lmijk,nijk,ijk -> nlm', angs_padded, rads, srho)
        coeff = []

        #remove zero padding from m
        for n in range(n_rad):
            for l in range(n_l):
                for m in range(2 * n_l + 1):
                    if abs(m - n_l) <= l:
                        coeff.append(coeff_array[n, l, m])

        return torch.stack(coeff).view(1, -1), angs

    def box_around(self, pos, radius):
        '''
        Return dictionary containing box around an atom at position pos with
        given radius. Dictionary contains box in mesh, euclidean and spherical
        coordinates

        Parameters
        ---

        Returns
        ---
            dict
                {'mesh','real','radial'}, box in mesh,
                euclidean and spherical coordinates
        '''
        pos = pos.view(-1)
        #Create box with max. distance = radius
        rmax = torch.ceil(radius / self.a)
        Xm, Ym, Zm = self.mesh_3d(self.U, self.a, scaled=False, rmax=rmax, indexing='ij')
        X, Y, Z = self.mesh_3d(self.U, self.a, scaled=True, rmax=rmax, indexing='ij')


        #Find mesh pos.
        cm = torch.round(self.U_inv.mv(pos))
        dr = pos - self.U.mv(cm)
        Xs = X - dr[0]
        Ys = Y - dr[1]
        Zs = Z - dr[2]

        Xm = torch.fmod((Xm + cm[0]), self.grid[0])
        Ym = torch.fmod((Ym + cm[1]), self.grid[1])
        Zm = torch.fmod((Zm + cm[2]), self.grid[2])

        R = torch.sqrt(Xs**2 + Ys**2 + Zs**2)

        Phi = torch.atan2(Ys, Xs)
        Theta = torch.acos(Zs/ R)
        Theta[R < 1e-15] = 0
        return {'mesh': [Xm, Ym, Zm], 'real': [Xs, Ys, Zs], 'radial': [R, Theta, Phi]}

    @staticmethod
    def mesh_3d(U, a, rmax, scaled=False, indexing='xy'):
        """
        Returns a 3d mesh taking into account periodic boundary conditions

        Parameters
        ----------

        rmax: list, int
            upper cutoff in every euclidean direction.
        scaled: boolean
            scale the meshes with unitcell size?
        indexing: 'xy' or 'ij'
            indexing scheme used by np.meshgrid.

        Returns
        -------

        X, Y, Z: tuple of np.ndarray
            defines mesh.
        """

        # resolve the periodic boundary conditions
        x_pbc = torch.cat([torch.arange(0, rmax[0] + 1), torch.arange(-rmax[0], 0)])
        y_pbc = torch.cat([torch.arange(0, rmax[1] + 1), torch.arange(-rmax[1], 0)])
        z_pbc = torch.cat([torch.arange(0, rmax[2] + 1), torch.arange(-rmax[2], 0)])

        Xm, Ym, Zm = torch.meshgrid([x_pbc, y_pbc, z_pbc])

        Rm = torch.cat([Xm.view(*Xm.shape, 1),
                        Ym.view(*Xm.shape, 1),
                        Zm.view(*Xm.shape, 1)], dim=3).double()

        if scaled:
            R = torch.einsum('ij,klmj -> iklm', U, Rm)
            X = R[0, :, :, :]
            Y = R[1, :, :, :]
            Z = R[2, :, :, :]
            return X, Y, Z
        else:
            return Xm.double(), Ym.double(), Zm.double()


class OrthoProjectorTorch(DefaultProjectorTorch, OrthoProjector):

    _registry_name = 'ortho_torch'

    @classmethod
    def dg(cls, r, basis, a):
        r_o = basis['r_o']
        return cls.dg_compiled(r, r_o, a)

    @staticmethod
    def dg_compiled(r, r_o, a):
        """
        Derivative of non-orthogonalized radial functions

        Parameters
        -------

            r: float
                radius
            basis: dict
                dictionary containing r_o
            a: int
                exponent (equiv. to radial index n)

        Returns
        ------

            float
                derivative of radial function at radius r
        """
        N = torch.sqrt(720*r_o**(11+2*a)*1/((2*a+11)*(2*a+10)*(2*a+9)*(2*a+8)*(2*a+7)*\
                                       (2*a+6)*(2*a+5)))
        return r * (r_o - r)**(a + 1) * (2 * r_o - (a + 4) * r) / N

    @staticmethod
    def g_compiled(r, r_o, a):
        """
        Non-orthogonalized radial functions

        Parameters
        -------

            r: float
                radius
            basis: dict
                dictionary containing r_o
            a: int
                exponent (equiv. to radial index n)

        Returns
        ------

            float
                value of radial function at radius r
        """
        N = torch.sqrt(720*r_o**(11+2*a)*1/((2*a+11)*(2*a+10)*(2*a+9)*(2*a+8)*(2*a+7)*\
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
    def S(cls, basis):
        '''
        Overlap matrix between radial basis functions

        Parameters
        -------

            r_o: float
                outer radial cutoff
            nmax: int
                max. number of radial functions

        Returns
        -------

            np.ndarray (nmax, nmax)
                Overlap matrix
        '''
        r_o = basis['r_o']
        nmax = basis['n']
        S_matrix = np.zeros([nmax, nmax])
        r_grid = np.linspace(0, r_o, 1000)
        dr = r_grid[1] - r_grid[0]
        for i in range(nmax):
            for j in range(i, nmax):
                S_matrix[i, j] = np.sum(OrthoProjector.g(r_grid, basis, i + 1) *\
                    OrthoProjector.g(r_grid, basis, j + 1) * r_grid**2) * dr
        for i in range(nmax):
            for j in range(i + 1, nmax):
                S_matrix[j, i] = S_matrix[i, j]
        return S_matrix

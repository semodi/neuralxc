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

        for species in basis_instructions:
            if len(species) < 3:
                W[species] = self.get_W(basis_instructions[species])

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

    def set_species(self, species):
        self.species = species

    def set_cell_parameters(self, unitcell, grid):

        a = torch.norm(unitcell, dim=1).double() / grid
        U = torch.einsum('ij,i->ij', unitcell, 1/grid)
        self.grid = grid
        self.V_cell = torch.abs(torch.det(U))
        self.U = torch.transpose(U,0,1)
        self.U_inv = torch.inverse(U)
        self.a = a


    def forward_basis(self, positions, unitcell, grid, my_box):
        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        box = self.box_around(positions, basis['r_o'], my_box)
        return self.get_basis_on_mesh(box, basis, self.W[self.species])


    def forward_fast(self, rho, positions, unitcell, grid, radials, angulars, my_box):
        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        Xm, Ym, Zm = self.mesh_around(positions, basis['r_o'], my_box)
        return  self.project_onto(rho[Xm,Ym,Zm], radials, angulars, int(basis['l']))



    def forward(self, rho, positions, species, unitcell, grid, a):

        self.set_cell_parameters(unitcell, grid, a)
        basis_rep = {}
        species = [str(element_dict[int(s.detach().numpy())]) for s in species]

        for pos, spec in zip(positions, species):
            if not spec in basis_rep:
                basis_rep[spec] = []

            basis = self.basis[spec]
            box = self.box_around(pos, basis['r_o'])
            projection = self.project(rho, box, basis, self.W[spec])

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

    def project_onto(self, rho, rads, angs, n_l):

        #zero_pad_angs (so that it can be converted to numpy array):
        # n_rad = rads.shape[0]
        # n_l = angs.shape[0]
        rads = rads * self.V_cell
        coeff_array = torch.einsum('lmijk,nijk,ijk -> nlm', angs, rads, rho)

        indexing = torch.ones(coeff_array.size()[1:])
        indexing = (torch.tril(indexing, diagonal = n_l) * torch.flip(torch.tril(indexing, diagonal = n_l), (1,))).bool()
        coeff = coeff_array[:, indexing]

        return coeff.view(-1)

    def get_basis_on_mesh(self, box, basis, W):

        n_rad = basis['n']
        n_l = basis['l']
        r_o = basis['r_o']
        R, Theta, Phi = box['radial']
        Xm, Ym, Zm = box['mesh']

        #Build angular part of basis functions
        angs = []
        for l in range(n_l):
            angs.append([])
            ang_l = self.angulars_real(l, Theta, Phi)
            for m in range(-l, l + 1):
                angs[l].append(ang_l[l + m])

        zeropad = torch.zeros_like(R)
        angs_padded = []
        for l in range(n_l):
            angs_padded.append(torch.stack([zeropad] * (n_l - l) + angs[l] + [zeropad] * (n_l - l)))

        angs_padded = torch.stack(angs_padded)

        rads = self.radials(R, basis, W)

        return rads, angs_padded


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

        #Build angular part of basis functions
        angs = []
        for l in range(n_l):
            angs.append([])
            ang_l = self.angulars_real(l, Theta, Phi)
            for m in range(-l, l + 1):
                angs[l].append(ang_l[l + m])

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

        return torch.stack(coeff).view(1, -1)

    def mesh_around(self, pos, radius, my_box):
        pos = pos.view(-1)
        #Create box with max. distance = radius
        rmax = torch.ceil(radius / self.a)
        cm = torch.round(self.U_inv.mv(pos))
        # my_box -= my_box[:,0].unsqueeze(1)
        Xm, Ym, Zm = self.mesh_3d(self.U, self.a, my_box = my_box,cm =cm, scaled=False, rmax=rmax, indexing='ij')
        #Find mesh pos.
        cm = shift(cm, self.grid)
        cm -= my_box[:,0]
        Xm = torch.fmod((Xm + cm[0]), self.grid[0])
        Ym = torch.fmod((Ym + cm[1]), self.grid[1])
        Zm = torch.fmod((Zm + cm[2]), self.grid[2])

        return Xm.long(), Ym.long(), Zm.long()

    def box_around(self, pos, radius, my_box):
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

        #Find mesh pos.
        cm = torch.round(self.U_inv.mv(pos))
        dr = pos - self.U.mv(cm)
        #Create box with max. distance = radius
        rmax = torch.ceil(radius / self.a)

        # my_box = my_box - cm.view(3,1)

        Xm, Ym, Zm = self.mesh_3d(self.U, self.a, my_box = my_box, cm = cm, scaled=False, rmax=rmax, indexing='ij')
        X, Y, Z = self.mesh_3d(self.U, self.a, my_box = my_box, cm = cm, scaled=True, rmax=rmax, indexing='ij')


        Xs = X - dr[0]
        Ys = Y - dr[1]
        Zs = Z - dr[2]

        #TODO: this could probably be done in 1d and moved to mesh_3d
        Xm = torch.fmod((Xm + cm[0]), self.grid[0])
        Ym = torch.fmod((Ym + cm[1]), self.grid[1])
        Zm = torch.fmod((Zm + cm[2]), self.grid[2])


        R = torch.sqrt(Xs**2 + Ys**2 + Zs**2)

        Phi = torch.atan2(Ys, Xs)
        Theta = torch.acos(Zs/ R)
        Theta[R < 1e-15] = 0
        return {'mesh': [Xm, Ym, Zm], 'real': [Xs, Ys, Zs], 'radial': [R, Theta, Phi]}

    def mesh_3d(self, U, a, rmax, my_box, cm, scaled=False, indexing='xy'):
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


        # x_pbc = torch.arange(-rmax[0], rmax[0] + 1, dtype=torch.float64)
        # y_pbc = torch.arange(-rmax[1], rmax[1] + 1, dtype=torch.float64)
        # z_pbc = torch.arange(-rmax[2], rmax[2] + 1, dtype=torch.float64)

        # Does the same as above but does not complain if unitcell.requires_gradient
        x_pbc = torch.arange(-1000, 1000, dtype=torch.float64)
        x_pbc = x_pbc[(x_pbc >= -rmax[0]) & (x_pbc <= rmax[0])]
        y_pbc = torch.arange(-1000, 1000, dtype=torch.float64)
        y_pbc = y_pbc[(y_pbc >= -rmax[1]) & (y_pbc <= rmax[1])]
        z_pbc = torch.arange(-1000, 1000, dtype=torch.float64)
        z_pbc = z_pbc[(z_pbc >= -rmax[2]) & (z_pbc <= rmax[2])]

        x_pbc_shifted = x_pbc + cm[0]
        y_pbc_shifted = y_pbc + cm[1]
        z_pbc_shifted = z_pbc + cm[2]

        # Shift all to positive, then resolve periodic boundary conditions to compare to myBox
        x_pbc_shifted = shift(x_pbc_shifted, self.grid[0])
        y_pbc_shifted = shift(y_pbc_shifted, self.grid[1])
        z_pbc_shifted = shift(z_pbc_shifted, self.grid[2])

        if not scaled:
            x_pbc = shift(x_pbc, self.grid[0])
            y_pbc = shift(y_pbc, self.grid[1])
            z_pbc = shift(z_pbc, self.grid[2])

        x_pbc = x_pbc[(x_pbc_shifted >= my_box[0,0]) & (x_pbc_shifted < my_box[0,1])]
        y_pbc = y_pbc[(y_pbc_shifted >= my_box[1,0]) & (y_pbc_shifted < my_box[1,1])]
        z_pbc = z_pbc[(z_pbc_shifted >= my_box[2,0]) & (z_pbc_shifted < my_box[2,1])]
        # print(x_pbc)
        #

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


def shift(c, g):
    c = torch.fmod(c + torch.ceil(torch.abs(torch.min(c)/g))*g, g)
    return c

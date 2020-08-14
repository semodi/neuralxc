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
try:
    import torch
    TorchModule = torch.nn.Module
except ModuleNotFoundError:
    class TorchModule:
         def __init__(self):
             pass


class DefaultProjectorTorch(TorchModule, BaseProjector) :

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
        TorchModule.__init__(self)
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
        # species = torch.Tensor([getattr(periodictable, s).number for s in species])
        my_box = torch.Tensor([[0, self.grid[i]] for i in range(3)])
        C = self.forward(rho, positions, species, self.unitcell, self.grid, my_box)
        return {spec: C[spec].detach().numpy() for spec in C}

    def set_species(self, species):
        self.species = species

    def set_cell_parameters(self, unitcell, grid):

        a = torch.norm(unitcell, dim=1).double() / grid
        U = torch.einsum('ij,i->ij', unitcell, 1/grid)
        self.grid = grid
        self.V_cell = torch.abs(torch.det(U))
        self.U = torch.transpose(U,0,1)
        self.U_inv = torch.inverse(self.U)
        self.a = a


    def forward_basis(self, positions, unitcell, grid, my_box):
        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        box, mesh = self.box_around(positions, basis['r_o'], my_box)
        rad, ang  =  self.get_basis_on_mesh(box, basis, self.W[self.species])
        return rad, ang, mesh


    def forward_fast(self, rho, positions, unitcell, grid, radials, angulars, my_box):
        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        Xm, Ym, Zm = my_box.long()
        return  self.project_onto(rho[Xm,Ym,Zm], radials, angulars, int(basis['l']))



    def forward(self, rho, positions, species, unitcell, grid, my_box):

        self.set_cell_parameters(unitcell, grid)
        basis_rep = {}
        # species = [str(element_dict[int(s.detach().numpy())]) for s in species]

        for pos, spec in zip(positions, species):
            if not spec in basis_rep:
                basis_rep[spec] = []

            self.species = spec
            rad, ang, mesh = self.forward_basis(pos, unitcell, grid, my_box)
            projection = self.forward_fast(rho,pos,unitcell, grid, rad, ang, mesh)
            basis_rep[spec].append(projection.view(1, -1))

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
        return res

    def project_onto(self, rho, rads, angs, n_l):

        rho = rho * self.V_cell
        if rho.ndim == 1:
            coeff_array = torch.einsum('li,ni,i -> nl', angs, rads, rho)
        else:
            coeff_array = torch.einsum('lmijk,nijk,ijk -> nlm', angs, rads, rho)

        return coeff_array.view(-1)

    def get_basis_on_mesh(self, box, basis, W):

        n_rad = basis['n']
        n_l = basis['l']
        r_o = basis['r_o']
        R, Theta, Phi = box['radial']

        #Build angular part of basis functions
        angs = []
        for l in range(n_l):
            angs += self.angulars_real(l, Theta, Phi)

        angs = torch.stack(angs)
        rads = self.radials(R, basis, W)

        return rads, angs



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
        rmax = torch.ceil(radius / self.a) + 2
        # my_box = my_box - cm.view(3,1)

        Xm = self.mesh_3d(self.U, self.a, my_box = my_box, cm = cm, scaled=False, rmax=rmax, indexing='ij')
        X = self.mesh_3d(self.U, self.a, my_box = my_box, cm = cm, scaled=True, rmax=rmax, indexing='ij')

        Xs = X - dr.view(-1,1,1,1)

        cms = shift(cm, self.grid)
        cms -= my_box[:,0]
        Xm = torch.fmod((Xm + cms.view(-1,1,1,1)), self.grid.view(-1,1,1,1))


        R = torch.norm(Xs, dim=0)

        co = (R <= radius)
        R = R[co]
        Xs = Xs[:, co]
        Xm = Xm[:, co]

        Phi = torch.atan2(Xs[1], Xs[0])
        Theta = torch.acos(Xs[2]/ R)
        Theta[R < 1e-15] = 0
        return {'radial': [R, Theta, Phi], 'co': co}, Xm

    def mesh_3d(self, U, a, rmax, my_box, cm, scaled=False, indexing='xy', both=False):
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

        Rm = torch.stack([Xm,
                        Ym,
                        Zm]).double()

        if scaled:
            R = torch.einsum('ij,jklm -> iklm', U, Rm)
            return R
        else:
            return Rm

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


class RadialProjectorTorch(OrthoProjectorTorch):

    _registry_name = 'ortho_radial_torch'

    def __init__(self, grid_coords, grid_weights, basis_instructions, **kwargs):
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
        TorchModule.__init__(self)
        self.basis = basis_instructions
        # Initialize the matrix used to orthonormalize radial basis
        W = {}
        for species in basis_instructions:
            if len(species) < 3:
                W[species] = self.get_W(basis_instructions[species])

        self.grid_coords = torch.from_numpy(grid_coords)
        self.grid_weights = torch.from_numpy(grid_weights)

        self.W = {w:torch.from_numpy(W[w]) for w in W}

        for species in basis_instructions:
            if len(species) < 3:
                W[species] = self.get_W(basis_instructions[species])

        self.my_box = torch.Tensor([[0, 1] for i in range(3)])
        self.unitcell = self.grid_coords
        self.grid = self.grid_weights

    def set_cell_parameters(self, grid_coords, grid_weights):
        self.grid_coords = grid_coords
        self.grid_weights = grid_weights
        self.V_cell = grid_weights


    def forward_fast(self, rho, positions, grid_coords, grid_weights, radials, angulars, my_box):
        self.set_cell_parameters(grid_coords, grid_weights)
        basis = self.basis[self.species]
        return  self.project_onto(rho, radials, angulars, int(basis['l']))


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

        # my_box = my_box - cm.view(3,1)

        Xm, Ym, Zm = torch.arange(self.grid_weights.size()[0]), None, None
        X, Y, Z = [self.grid_coords[:,i] - pos[i] for i in range(3)]

        R = torch.sqrt(X**2 + Y**2 + Z**2)

        Phi = torch.atan2(Y, X)
        Theta = torch.acos(Z/ R)
        Theta[R < 1e-15] = 0
        return {'mesh': [Xm, Ym, Zm], 'real': [X, Y, Z], 'radial': [R, Theta, Phi]}

def shift(c, g):
    c = torch.fmod(c + torch.ceil(torch.abs(torch.min(c)/g))*g, g)
    return c

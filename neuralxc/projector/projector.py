"""
projector.py
Defines the two base classes from which all other projectors are derived:
EuclideanProjector for euclidean grids with periodic boundary conditions and
RadialProjector for generalized grids without PBCs.
"""

from abc import abstractmethod

import numpy as np
import torch
from opt_einsum import contract
from torch.nn import Module as TorchModule

from neuralxc.base import ABCRegistry
from neuralxc.utils import geom


class ProjectorRegistry(ABCRegistry):
    REGISTRY = {}


def DensityProjector(**kwargs):

    basis_instructions = kwargs['basis_instructions']
    application = basis_instructions.get('application', 'siesta')
    projector_type = basis_instructions.get('projector_type', 'ortho')
    if application == 'pyscf' and projector_type == 'ortho':
        projector_type = 'pyscf'

    registry = BaseProjector.get_registry()
    if not projector_type in registry:
        raise Exception('Projector: {} not registered'.format(projector_type))

    return registry[projector_type](**kwargs)


class BaseProjector(TorchModule, metaclass=ProjectorRegistry):
    _registry_name = 'base'
    _unit_test = False

    def __init__(self):
        TorchModule.__init__(self)

    @abstractmethod
    def get_basis_rep(self):
        pass

    def get_basis_rep(self, rho, positions, species, **kwargs):
        """Calculates the basis representation for a given real space density

        Parameters
        ------------------
        rho: np.ndarray float (npoints) or (xpoints, ypoints, zpoints)
        	Electron density in real space
        positions: np.ndarray float (natoms, 3)
        	atomic positions
        species: list string
        	atomic species (chem. symbols)

        Returns
        ------------
        c: dict of np.ndarrays
        	Basis representation, dict keys correspond to atomic species.
        """
        rho = torch.from_numpy(rho)
        positions = torch.from_numpy(positions)
        my_box = torch.Tensor([[0, self.grid[i]] for i in range(3)])
        C = self.forward(rho, positions, species, self.unitcell, self.grid, my_box)
        return {spec: C[spec].detach().numpy() for spec in C}

    def set_species(self, species):
        self.species = species

    def forward(self, rho, positions, species, unitcell, grid, my_box):
        """ Combines basis set creation (done in forward_basis) and projection
        (done in forward_fast)

        Parameters
        ----------
        rho, Tensor (npoints) or (xpoints, ypoints, zpoints)
            electron density on grid
        positions, Tensor (natoms, 3)
        	atomic position
        species, list of str
            atomic species (chem. symbols)
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
        basis_rep = {}
        # species = [str(element_dict[int(s.detach().numpy())]) for s in species]
        # if rho.dim() == 4:
        #     rho = rho.permute(1, 2, 3, 0)
        # elif rho.dim() == 2:
        #     rho = rho.permute(1,0)

        for pos, spec in zip(positions, species):
            if not spec in basis_rep:
                basis_rep[spec] = []

            self.species = spec
            rad, ang, mesh = self.forward_basis(pos, unitcell, grid, my_box)
            projection = self.forward_fast(rho, pos, unitcell, grid, rad, ang, mesh)
            basis_rep[spec].append(projection.view(1, -1))

        for spec in basis_rep:
            basis_rep[spec] = torch.cat(basis_rep[spec], dim=0)

        return basis_rep

    @staticmethod
    def angulars_real(l, theta, phi):
        """ Spherical harmonics (uses physics convention for angles)

        Parameters
        ----------
        l: int
            angular momentum quantum number

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
        for m in range(-l, l + 1):
            res.append(geom.SH(l, m, theta, phi))
        return res


class EuclideanProjector(BaseProjector):

    _registry_name = 'euclidean'
    _unit_test = False

    def __init__(self, unitcell, grid, basis_instructions, **kwargs):
        """
        Projector on euclidean grid with periodic bounday conditions

        Parameters
        ------------------
        unitcell: numpy.ndarray float (3,3)
        	Unitcell in bohr
        grid: numpy.ndarray float (3)
        	Grid points per unitcell
        basis_instructions: dict
        	Instructions that define basis
        """
        super().__init__()
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
        self.W = {w: torch.from_numpy(W[w]) for w in W}

        for species in basis_instructions:
            if len(species) < 3:
                W[species] = self.get_W(basis_instructions[species])

    def set_cell_parameters(self, unitcell, grid):
        a = torch.norm(unitcell, dim=1).double() / grid
        U = contract('ij,i->ij', unitcell, 1 / grid)
        self.grid = grid
        self.V_cell = torch.abs(torch.det(U))
        self.U = torch.transpose(U, 0, 1)
        self.U_inv = torch.inverse(self.U)
        self.a = a

    def forward_fast(self, rho, positions, unitcell, grid, radials, angulars, mesh):
        """Creates basis set (for projection) for a single atom, on grid points

        Parameters
        ----------
        rho, Tensor (npoints) or (xpoints, ypoints, zpoints)
            electron density on grid
        positions, Tensor (1, 3) or (3)
        	atomic position
        unitcell, Tensor (3,3)
        	Unitcell in bohr
        grid, Tensor (3)
        	Grid points per unitcell
        radials, Tensor ()
            Radial functions on grid, stacked
        angular, Tensor ()
            Angular functions on grid, stacked
        mesh, Tensor (3: euclid. directions, npoints)
            3d meshgrid

        Returns
        --------
        rad, ang, mesh
            Stacked radial and angular functions as well as meshgrid
        """
        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        Xm, Ym, Zm = mesh.long()
        return self.project_onto(rho[..., Xm, Ym, Zm], radials, angulars, int(basis['l']))

    def box_around(self, pos, radius, my_box):
        '''
        Create box around an atom with given cutoff-radius.

        Parameters
        ---
            pos, Tensor (1, 3) or (3)
            	atomic position
            radius, float
                cutoff radius in Bohr
            my_box, Tensor (3: euclid. directions, 2: upper and lower limits)
                Limiting box local gridpoints. Relevant if global grid is decomposed
                with MPI or similar.
        Returns
        ---
            tuple
                {'radial','co'}, mesh
                'radial': spherical coordinates,
                'co': cutoff filter
                mesh: meshgrid for indexing rho
        '''
        pos = pos.view(-1)

        #Find mesh pos.
        cm = torch.round(self.U_inv.mv(pos))
        dr = pos - self.U.mv(cm)
        #Create box with max. distance = radius
        rmax = torch.ceil(radius / self.a) + 2
        # my_box = my_box - cm.view(3,1)

        Xm = self.mesh_3d(self.U, self.a, my_box=my_box, cm=cm, scaled=False, rmax=rmax, indexing='ij')
        X = self.mesh_3d(self.U, self.a, my_box=my_box, cm=cm, scaled=True, rmax=rmax, indexing='ij')

        Xs = X - dr.view(-1, 1, 1, 1)

        cms = shift(cm, self.grid)
        cms -= my_box[:, 0]
        Xm = torch.fmod((Xm + cms.view(-1, 1, 1, 1)), self.grid.view(-1, 1, 1, 1))

        R = torch.norm(Xs, dim=0)

        co = (R <= radius)
        R = R[co]
        Xs = Xs[:, co]
        Xm = Xm[:, co]

        Phi = torch.atan2(Xs[1], Xs[0])
        Theta = torch.acos(Xs[2] / R)
        Theta[R < 1e-15] = 0
        return {'radial': [R, Theta, Phi], 'co': co}, Xm

    def mesh_3d(self, U, a, rmax, my_box, cm, scaled=False, indexing='xy', both=False):
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

        x_pbc = x_pbc[(x_pbc_shifted >= my_box[0, 0]) & (x_pbc_shifted < my_box[0, 1])]
        y_pbc = y_pbc[(y_pbc_shifted >= my_box[1, 0]) & (y_pbc_shifted < my_box[1, 1])]
        z_pbc = z_pbc[(z_pbc_shifted >= my_box[2, 0]) & (z_pbc_shifted < my_box[2, 1])]

        Xm, Ym, Zm = torch.meshgrid([x_pbc, y_pbc, z_pbc])

        Rm = torch.stack([Xm, Ym, Zm]).double()

        if scaled:
            R = contract('ij,jklm -> iklm', U, Rm)
            return R
        else:
            return Rm


class RadialProjector(BaseProjector):

    _registry_name = 'radial'

    def __init__(self, grid_coords, grid_weights, basis_instructions, **kwargs):
        """
        Projector for generalized grid (as provided by e.g. PySCF). More flexible
        than euclidean grid as only grid point coordinates and their integration
        weights need to be provided, however does not support periodic boundary
        conditions. Special use case: Radial grids, as used by all-electron codes.

        Parameters
        ------------------
        grid_coords: numpy.ndarray (npoints, 3)
        	Coordinates of radial grid points
        grid_weights: numpy.ndarray (npoints)
        	Grid weights for integration
        basis_instructions: dict
        	Instructions that defines basis
        """
        BaseProjector.__init__(self)
        self.basis = basis_instructions
        # Initialize the matrix used to orthonormalize radial basis
        W = {}
        for species in basis_instructions:
            if len(species) < 3:
                W[species] = self.get_W(basis_instructions[species])

        self.grid_coords = torch.from_numpy(grid_coords)
        self.grid_weights = torch.from_numpy(grid_weights)

        self.W = {w: torch.from_numpy(W[w]) for w in W}

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
        Xm = my_box.long()
        grid_weights = grid_weights[Xm]
        self.set_cell_parameters(grid_coords, grid_weights)
        basis = self.basis[self.species]
        return self.project_onto(rho[..., Xm], radials, angulars, int(basis['l']))

    def box_around(self, pos, radius, my_box=None):
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

        # Create box with max. distance = radius
        Xm = torch.arange(self.grid_weights.shape[0])
        X = (self.grid_coords - pos.view(-1, 3)).T

        R = torch.norm(X, dim=0)

        co = (R <= radius)
        R = R[co]
        X = X[:, co]
        Xm = Xm[co]

        Phi = torch.atan2(X[1], X[0])

        Theta = torch.acos(X[2] / R)
        Theta[R < 1e-15] = 0
        return {'radial': [R, Theta, Phi], 'co': co}, Xm.view(1, -1)


def shift(c, g):
    c = torch.fmod(c + torch.ceil(torch.abs(torch.min(c) / g)) * g, g)
    return c

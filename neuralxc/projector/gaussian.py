"""
gaussian.py
Implements density projection basis with radial functions based on Gaussians.
Gaussians can be dampened (gamma) and truncated (r_o) for integration on numerical grids.
"""
import os

import numpy as np
import pyscf.gto as gto
import pyscf.gto.basis as gtobasis
import torch
from opt_einsum import contract
from torch.nn import Module as TorchModule

from neuralxc.projector import (BaseProjector, EuclideanProjector, RadialProjector)
from neuralxc.pyscf import BasisPadder

# Normalization factors
GAMMA = torch.from_numpy(
    np.array([1 / 2, 3 / 4, 15 / 8, 105 / 16, 945 / 32, 10395 / 64, 135135 / 128]) * np.sqrt(np.pi))


def parse_basis(basis_instructions):
    full_basis = {}
    basis_strings = {}
    for species in basis_instructions:
        if len(species) < 3:
            if os.path.isfile(basis_instructions[species]['basis']):
                basis_strings[species] = open(basis_instructions[species]['basis'], 'r').read()
                bas = gtobasis.parse(basis_strings[species])
            else:
                basis_strings[species] = basis_instructions[species]['basis']
                bas = basis_strings[species]

            spec = 'O' if species == 'X' else species
            try:
                mol = gto.M(atom='{} 0 0 0'.format(spec), basis={spec: bas})
            except RuntimeError:
                mol = gto.M(atom='{} 0 0 0'.format(spec), basis={spec: bas}, spin=1)
            sigma = basis_instructions[species].get('sigma', 2.0)
            gamma = basis_instructions[species].get('gamma', 1.0)
            basis = {}
            for bi in range(mol.atom_nshells(0)):
                l = mol.bas_angular(bi)
                if l not in basis:
                    basis[l] = {'alpha': [], 'r_o': [], 'coeff': [], 'gamma': []}
                alpha = mol.bas_exp(bi)
                coeff = mol.bas_ctr_coeff(bi)
                r_o = alpha**(-1 / 2) * sigma * (1 + l / 5)
                basis[l]['alpha'].append(alpha)
                basis[l]['r_o'].append(r_o)
                basis[l]['gamma'].append(gamma)
                basis[l]['coeff'].append(coeff)
            basis = [{
                'l': l,
                'alpha': basis[l]['alpha'],
                'r_o': basis[l]['r_o'],
                'gamma': basis[l]['gamma'],
                'coeff': basis[l]['coeff']
            } for l in basis]
            full_basis[species] = basis
    return full_basis, basis_strings


class GaussianProjectorMixin():
    """Implements GTO basis
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
            Stacked radial and angular functions as well as meshgrid stacked
            with grid in spherical coordinates
        """
        r_o_max = np.max([np.max(b['r_o']) for b in self.basis[self.species]])

        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        box, mesh = self.box_around(positions, r_o_max, my_box)
        box['mesh'] = mesh
        rad, ang = self.get_basis_on_mesh(box, basis)
        return rad, ang, torch.cat([mesh.double(), box['radial']])

    def get_basis_on_mesh(self, box, basis_instructions):

        angs = []
        rads = []

        box['radial'] = torch.stack(box['radial'])
        for ib, basis in enumerate(basis_instructions):
            l = basis['l']
            # r_o_max = np.max(basis['r_o'])
            # filt = (box['radial'][0] <= r_o_max)
            filt = (box['radial'][0] <= 1000000)
            box_rad = box['radial'][:, filt]
            # box_m = box['mesh'][:, filt]
            # ang = torch.zeros([2 * l + 1, filt.size()[0]], dtype=torch.double)
            # rad = torch.zeros([len(basis['r_o']), filt.size()[0]], dtype=torch.double)
            # ang[:,filt] = torch.stack(self.angulars_real(l, box_rad[1], box_rad[2])) # shape (m, x, y, z)
            # rad[:,filt] = torch.stack(self.radials(box_rad[0], [basis])[0]) # shape (n, x, y, z)
            # rads.append(rad)
            # angs.append(ang)
            angs.append(torch.stack(self.angulars_real(l, box_rad[1], box_rad[2])))  # shape (m, x, y, z)
            rads.append(torch.stack(self.radials(box_rad[0], [basis])[0]))  # shape (n, x, y, z)

        return torch.cat(rads), torch.cat(angs)

    def project_onto(self, rho, rads, angs, basis_instructions, basis_string, box):

        rad_cnt = 0
        ang_cnt = 0
        coeff = []
        for basis in basis_instructions:
            # print(basis)
            l = basis['l']
            len_rad = len(basis['r_o'])
            rad = rads[rad_cnt:rad_cnt + len_rad]
            ang = angs[ang_cnt:ang_cnt + (2 * l + 1)]
            rad_cnt += len_rad
            ang_cnt += 2 * l + 1
            r_o_max = np.max(basis['r_o'])
            # filt = (box['radial'][0] <= r_o_max)
            # filt = (box['radial'][0] <= 1000000)
            # rad *= self.V_cell
            # coeff.append(contract('i,mi,ni -> nm', rho[filt], ang[:,filt], rad[:,filt]).reshape(-1))
            coeff.append(contract('i,mi,ni -> nm', rho, ang, rad*self.V_cell).reshape(-1))

        coeff = torch.cat(coeff)

        coeff_out = torch.mv(self.M[self.species], coeff)
        return coeff_out

    def init_padder(self, basis_instructions):
        basis_strings = self.basis_strings
        self.M = {}
        self.symmetrize_instructions = {'basis': {}}
        for species in basis_instructions:
            if len(species) < 3:
                try:
                    bas = gtobasis.parse(basis_strings[species])
                except:
                    bas = basis_strings[species]
                spec = 'O' if species == 'X' else species
                try:
                    mol = gto.M(atom='{} 0 0 0'.format(spec), basis={spec: bas})
                except RuntimeError:
                    mol = gto.M(atom='{} 0 0 0'.format(spec), basis={spec: bas}, spin=1)
                bp = BasisPadder(mol)
                il = bp.indexing_l[spec][0]
                ir = bp.indexing_r[spec][0]
                M = np.zeros([len(il), len(ir)])
                M[il, ir] = 1
                self.M[species] = torch.from_numpy(M).double()
                self.symmetrize_instructions['basis'].update(bp.get_basis_json())

    @classmethod
    def g(cls, r, r_o, alpha, l, gamma):
        fc = 1 - (.5 * (1 - torch.cos(np.pi * (r / gamma) / r_o[0])))**8
        N = (2 * alpha[0])**(l / 2 + 3 / 4) * np.sqrt(2) / np.sqrt(GAMMA[l])
        f = (r / gamma)**l * torch.exp(-alpha[0] * (r / gamma)**2) * fc * N
        f[(r / gamma) > r_o[0]] = 0
        return f

    @classmethod
    def get_W(cls, basis):
        return np.eye(3)

    @classmethod
    def radials(cls, r, basis, W=None):
        result = []
        if isinstance(basis, list):
            for b in basis:
                res = []
                for ib, alpha in enumerate(b['alpha']):
                    res.append(cls.g(r, b['r_o'][ib], b['alpha'][ib], b['l'], b['gamma'][ib]))
                result.append(res)
        elif isinstance(basis, dict):
            result.append([cls.g(r, basis['r_o'], basis['alpha'], basis['l'], basis['gamma'])])
        return result


class GaussianEuclideanProjector(EuclideanProjector, GaussianProjectorMixin):
    """Implements GTO basis

    :_registry_name: 'gaussian'
    """

    _registry_name = 'gaussian'
    _unit_test = True

    def __init__(self, unitcell, grid, basis_instructions, **kwargs):
        """Implements GTO basis on euclidean grid

        Parameters
        ------------------
        unitcell, numpy.ndarray float (3,3)
        	Unitcell in bohr
        grid, numpy.ndarray float (3)
        	Grid points per unitcell
        basis_instructions, dict
        	Instructions that define basis
        """
        full_basis, basis_strings = parse_basis(basis_instructions)
        basis = {key: val for key, val in basis_instructions.items()}
        basis.update(full_basis)
        self.basis_strings = basis_strings
        EuclideanProjector.__init__(self, unitcell, grid, basis, **kwargs)
        self.init_padder(basis_instructions)

    def forward_fast(self, rho, positions, unitcell, grid, radials, angulars, my_box):
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
        angulars, Tensor ()
            Angular functions on grid, stacked
        my_box, Tensor (6, npoints)
            (:3,:) 3d meshgrid
            (3:,:) 3d spherical grid

        Returns
        --------
        rad, ang, mesh
            Stacked radial and angular functions as well as meshgrid
        """
        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        box = {}
        box['mesh'] = my_box[:3]
        box['radial'] = my_box[3:]
        Xm, Ym, Zm = box['mesh'].long()
        return self.project_onto(rho[Xm, Ym, Zm], radials, angulars, basis, self.basis_strings[self.species], box)


class GaussianRadialProjector(RadialProjector, GaussianProjectorMixin):
    """
    :_registry_name: 'gaussian_radial'
    """
    _registry_name = 'gaussian_radial'
    _unit_test = False

    def __init__(self, grid_coords, grid_weights, basis_instructions, **kwargs):
        """Implements GTO basis on radial grid

        Parameters
        ------------------
        grid_coords, numpy.ndarray (npoints, 3)
        	Coordinates of radial grid points
        grid_weights, numpy.ndarray (npoints)
        	Grid weights for integration
        basis_instructions, dict
        	Instructions that defines basis
        """
        RadialProjector.__init__(self, grid_coords, grid_weights, basis_instructions, **kwargs)

        self.grid_coords = torch.from_numpy(grid_coords)
        self.grid_weights = torch.from_numpy(grid_weights)
        self.V_cell = self.grid_weights
        full_basis, basis_strings = parse_basis(basis_instructions)
        basis = {key: val for key, val in basis_instructions.items()}
        basis.update(full_basis)
        self.basis_strings = basis_strings
        self.basis = basis
        self.all_angs = {}
        self.unitcell = self.grid_coords
        self.grid = self.grid_weights
        self.init_padder(basis_instructions)

    def forward_fast(self, rho, positions, grid_coords, grid_weights, radials, angulars, my_box):
        self.set_cell_parameters(grid_coords, grid_weights)
        basis = self.basis[self.species]
        box = {}
        box['mesh'] = my_box[0]
        box['radial'] = my_box[1:]
        Xm = box['mesh'].long()
        return self.project_onto(rho[Xm], radials, angulars, basis, self.basis_strings[self.species], box)

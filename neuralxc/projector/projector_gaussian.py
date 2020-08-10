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
from ..timer import timer
import neuralxc.config as config
from ..utils import geom
import pyscf.gto.basis as gtobasis
import pyscf.gto as gto
from .projector import OrthoProjector, DefaultProjector,BaseProjector, RadialProjector
import neuralxc

GAMMA = np.array([1/2,3/4,15/8,105/16,945/32,10395/64,135135/128])*np.sqrt(np.pi)
class GaussianProjector(DefaultProjector):

    _registry_name = 'gaussian'

    def __init__(self, unitcell, grid, basis_instructions, **kwargs):

        full_basis, basis_strings = parse_basis(basis_instructions)
        basis = {key:val for key,val in basis_instructions.items()}
        basis.update(full_basis)
        self.basis_strings = basis_strings
        DefaultProjector.__init__(self, unitcell, grid, basis, **kwargs)

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
        basis_rep = {}
        for pos, spec in zip(positions, species):
            if not spec in basis_rep:
                basis_rep[spec] = []

            idx = '{}{}{}{}'.format(spec, pos[0], pos[1], pos[2])
            basis = self.basis[spec]
            # box = self.box_around(pos, basis['r_o'])
            projection, angs = self.project(rho,pos, basis,self.basis_strings[spec], angs=self.all_angs.get(idx, None))
            basis_rep[spec].append(projection)
            if config.UseMemory:
                self.all_angs[idx] = angs

        for spec in basis_rep:
            basis_rep[spec] = np.concatenate(basis_rep[spec], axis=0)

        return basis_rep


    def project(self, rho, pos, basis_instructions, basis_string, angs=None):
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

        coeff = []
        for ib, basis in enumerate(basis_instructions):
            l = basis['l']
            box = self.box_around(pos, np.max(basis['r_o']))
            ang = self.angulars_real(l, *box['radial'][1:]) # shape (m, x, y, z)
            rad = np.stack(self.radials(box['radial'][0], [basis])[0]) # shape (n, x, y, z)
            rad *= self.V_cell
            if rho.ndim == 1:
                c = np.einsum('i,mi,ni -> nm', rho, ang, rad, optimize=True)
            else:
                c = np.einsum('ijk,mijk,nijk -> nm',rho[box['mesh']], ang, rad, optimize=True)
            coeff += c.flatten().tolist()

        mol = gto.M(atom='O 0 0 0',
                    basis={'O': gtobasis.parse(basis_string)})

        bp = neuralxc.pyscf.BasisPadder(mol)
        coeff = bp.pad_basis(np.array(coeff))['O']
        return np.array(coeff).reshape(1, -1), None


    @classmethod
    def g(cls, r, r_o, alpha, l):
        fc = 1-(.5*(1-np.cos(np.pi*r/r_o)))**8
        N = (2*alpha)**(l/2+3/4)*np.sqrt(2)/np.sqrt(GAMMA[l])
        f = r**l*np.exp(-alpha*r**2)*fc*N
        f[r>r_o] = 0
        return f

    @classmethod
    def get_W(cls, basis):
        return np.eye(3)

    @classmethod
    def radials(cls, r, basis, W = None):
        result = []
        for b in basis:
            res = []
            for ib, alpha in enumerate(b['alpha']):
                res.append(cls.g(r, b['r_o'][ib], b['alpha'][ib], b['l']))
            result.append(res)
        return result

class RadialGaussianProjector(GaussianProjector):

    _registry_name = 'gaussian_radial'


    def __init__(self, basis_instructions, grid_coords, grid_weights, **kwargs):
        self.grid_coords = grid_coords
        self.grid_weights = grid_weights
        self.V_cell = self.grid_weights
        full_basis, basis_strings = parse_basis(basis_instructions)
        basis = {key:val for key,val in basis_instructions.items()}
        basis.update(full_basis)
        self.basis_strings = basis_strings
        self.basis = basis
        self.all_angs = {}

    box_around = RadialProjector.box_around

def parse_basis(basis_instructions):
    full_basis = {}
    basis_strings = {}
    for species in basis_instructions:
        if len(species) < 3:
            basis_strings[species] = open(basis_instructions[species]['basis'],'r').read()
            bas = gtobasis.parse(basis_strings[species])
            sigma = basis_instructions[species].get('sigma',2.0)
            basis = []
            for b in bas:
                l = b[0]
                alpha = np.array(b[1:])[:,0]
                r_o = alpha**(-1/2)*sigma*(1+l/5)
                basis.append({'l':l,'alpha':alpha,'r_o':r_o})
            full_basis[species] = basis
    return full_basis, basis_strings

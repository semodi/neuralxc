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

class ProjectorRegistry(ABCRegistry):
    REGISTRY = {}


class BaseProjector(metaclass=ProjectorRegistry):

    _registry_name = 'base'

    @abstractmethod
    def __init__(self, unitcell, grid, basis_instructions):
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
        pass

    @abstractmethod
    def get_basis_rep(self, rho, positions, species):
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
        pass

    @abstractmethod
    def get_V(self, dEdC, positions, species, calc_forces):
        """Calculates the basis representation for a given real space density

        Parameters
        ------------------
        dEdc , dict of numpy.ndarray

        positions, array float
        	atomic positions
        species, list string
        	atomic species (chem. symbols)
        calc_forces, bool
        	Calc. and return force corrections + stress corrections (concatenated)
        Returns
        ------------
        V, (force_correction) np.ndarray
        """
        pass


def DensityProjector(**kwargs):

    basis_instructions = kwargs['basis_instructions']
    application = basis_instructions.get('application', 'siesta')
    projector_type = basis_instructions.get('projector_type', 'ortho')
    if application == 'pyscf':
        projector_type = 'pyscf'

    registry = BaseProjector.get_registry()
    if not projector_type in registry:
        raise Exception('Projector: {} not registered'.format(projector_type))

    return registry[projector_type](**kwargs)


class DefaultProjector(BaseProjector):

    _registry_name = 'default'

    #TODO: Make some functions private
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
        self.basis = basis_instructions

        # Initialize the matrix used to orthonormalize radial basis
        W = {}
        for species in basis_instructions:
            if len(species) < 3:
                W[species] = self.get_W(basis_instructions[species])

        # Determine unitcell constants
        U = np.array(unitcell)  # Matrix to go from real space to mesh coordinates
        for i in range(3):
            U[i, :] = U[i, :] / grid[i]
        a = np.linalg.norm(unitcell, axis=1) / grid[:3]

        self.unitcell = unitcell
        self.grid = grid
        self.V_cell = np.linalg.det(U)
        self.U = U.T
        self.U_inv = np.linalg.inv(U)
        self.a = a
        self.W = W
        self.all_angs = {}

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
            box = self.box_around(pos, basis['r_o'])
            projection, angs = self.project(rho, box, basis, self.W[spec], angs=self.all_angs.get(idx, None))

            basis_rep[spec].append(projection)
            if config.UseMemory:
                self.all_angs[idx] = angs

        for spec in basis_rep:
            basis_rep[spec] = np.concatenate(basis_rep[spec], axis=0)

        return basis_rep

    def get_basis_rep_dict(self, rho, positions, species):
        """ Same as get_basis_rep but return feature as a dict with keys
        corresponding to quantum numbers, like: {'n,l,m' : feature}
        """
        basis_rep = {}
        for pos, spec in zip(positions, species):
            if not spec in basis_rep:
                basis_rep[spec] = []

            basis = self.basis[spec]
            box = self.box_around(pos, basis['r_o'])

            basis_rep[spec].append(self.project(rho, box, basis, self.W[spec], True))

            basis_rep[spec].append(projection)
            self.all_angs.append(angs)

        return basis_rep

    def get_V(self, dEdC, positions, species, calc_forces=False, rho=None, **kwargs):
        """Calculates the basis representation for a given real space density

        Parameters
        ------------------
        dEdc , dict of numpy.ndarray

        positions, array float
        	atomic positions
        species, list string
        	atomic species (chem. symbols)
        calc_forces, bool
        	Calc. and return force corrections + stress corrections (concatenated)
        Returns
        ------------
        V, (force_correction) np.ndarray
        """
        if isinstance(dEdC, list):
            dEdC = dEdC[0]

        V = np.zeros(self.grid)
        spec_idx = {spec: -1 for spec in species}
        force_corrections = np.zeros([len(species), 3])
        for i, (pos, spec) in enumerate(zip(positions, species)):
            spec_idx[spec] += 1
            if dEdC[spec].ndim == 3:
                assert dEdC[spec].shape[0] == 1
                dEdC[spec] = dEdC[spec][0]

            coeffs = dEdC[spec][spec_idx[spec]]
            basis = self.basis[spec]
            box = self.box_around(pos, basis['r_o'])

            idx = '{}{}{}{}'.format(spec, pos[0], pos[1], pos[2])
            V[tuple(box['mesh'])] += self.build(coeffs, box, basis, self.W[spec], angs=self.all_angs.get(idx, None))
            if calc_forces:
                if not isinstance(rho, np.ndarray):
                    raise ValueError('Must provide rho as np.ndarray')
                force_corrections[i] = self.get_force_correction(rho,
                                                                 coeffs,
                                                                 box,
                                                                 basis,
                                                                 self.W[spec],
                                                                 angs=self.all_angs.get(idx, None))

        stress_correction = np.einsum('ij,ik-> jk', force_corrections, positions)

        force_corrections = np.concatenate([force_corrections, stress_correction], axis=0)
        if calc_forces:
            return V, force_corrections
        else:
            return V

    def angulars(self, l, m, theta, phi):
        """ Angular functions (uses physics convention for angles)
        (For compatibility with complex version)
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
        # assert False
        # return sph_harm(m, l, phi, theta)

        angulars = self.angulars_real(l, theta, phi)
        return angulars[m + l]

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
        if l > 0:
            M = M_make_complex(l + 1)[-(2 * l + 1):, -(2 * l + 1):]
            M = np.linalg.inv(M)
        else:
            M = np.eye(1)

        shape = (2 * l + 1, ) + phi.shape
        sh_list = np.zeros(shape, dtype=np.complex)
        for i, m in enumerate(range(-l, l + 1)):
            sh_list[i] = sph_harm(m, l, phi, theta)

        ang = np.einsum('ij,j...-> i...', M, sh_list)
        return ang.real

    def get_force_correction(self, rho, coeffs, box, basis, W=None, angs=None):
        """ Calculate the contribution to the forces that arises from the
        dependence of the (nxc-)basis set on the atomic positions

        Parameters
        ----------
            rho: np.ndarray
                electron charge density on grid
            coeffs: list of floats
                coefficients dEdC for orbitals belonging to this atom
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
        -------
            force: np.ndarray
                force correction
        """

        n_rad = basis['n']
        n_l = basis['l']
        r_o = basis['r_o']

        R, Theta, Phi = box['radial']
        Xm, Ym, Zm = box['mesh']
        X, Y, Z = box['real']

        #Build angular part of basis functions
        if not isinstance(angs, list):
            angs = []
            for l in range(n_l):
                angs.append([])
                ang_l = self.angulars_real(l, Theta, Phi)
                for m in range(-l, l + 1):
                    # angs[l].append(sph_harm(m, l, Phi, Theta).conj()) TODO: In theory should be conj!?
                    # angs[l].append(self.angulars(l, m, Theta, Phi))
                    angs[l].append(ang_l[l + m])

        timer.start('force:basis_functions:dangular')
        # Derivatives of spherical harmonic
        # M = M_make_complex(n_l)
        dangs = np.zeros([3, n_l**2, len(X.flatten())])

        for ir, r in enumerate(zip(X.flatten(), Y.flatten(), Z.flatten())):
            dangs[:, :, ir] = grlylm(n_l - 1, r)  # shape: (3, n_l*n_l)

        # dangs = np.einsum('ij,kjl -> ilk', M, dangs, optimize=True)
        dangs = dangs.transpose((1, 2, 0))
        # print(dangs.shape)
        dangs = dangs.reshape(len(dangs), *X.shape, 3)

        timer.stop('force:basis_functions:dangular')
        timer.start('force:basis_functions:radial')
        #Build radial part of b.f.
        if not isinstance(W, np.ndarray):
            W = self.get_W(basis)  # Matrix to orthogonalize radial basis

        drads = self.dradials(R, basis, W)
        rads = self.radials(R, basis, W)
        radsr = np.array(rads)
        # radsr[R<1e-15] = 0
        radsr = radsr / R
        # radsr[:, R < 1e-15] = 0

        rhat = np.array([X / R, Y / R, Z / R])
        # rhat[:, R < 1e-15] = 0
        rho = rho[tuple(box['mesh'])]
        force = np.zeros(3)

        timer.stop('force:basis_functions:radial')

        timer.start('force:integrals')

        timer.start('force:integrals:precomp')
        # Precomputing common factors in loops below (saves factor 3 comp. time)
        RtoL = [R**l for l in range(n_l)]
        radbyrl = [[rads[n] / RtoL[l] for l in range(n_l)] for n in range(n_rad)]
        vmult0 = [[(drads[n] - l * radsr[n]) for l in range(n_l)] for n in range(n_rad)]
        vmult = [[[angs[l][m] * vmult0[n][l] for m in range(2 * l + 1)] for l in range(n_l)] for n in range(n_rad)]
        timer.stop('force:integrals:precomp')

        for ix in range(3):
            v = np.zeros_like(rho)
            idx_coeff = 0
            for n in range(n_rad):
                idx_l = 0
                for l in range(n_l):
                    for m in range(2 * l + 1):
                        v2 = (radbyrl[n][l] * dangs[idx_l, :, :, :, ix])
                        v += coeffs[idx_coeff] * ((vmult[n][l][m] * rhat[ix]) + v2)
                        idx_l += 1
                        idx_coeff += 1

            # assert np.allclose(v.imag, np.zeros_like(v))
            v[R < 1e-15] = 0
            force[ix] = np.sum(rho * v) * self.V_cell
        timer.stop('force:integrals')
        return force

    def build(self, coeffs, box, basis, W=None, angs=None):
        """ Build the contribution from this atom to the potential V in a
        provided bounding box

        Parameters
        ----------
            coeffs: list of floats
                coefficients dEdC for orbitals belonging to this atom
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
        -------
            V: np.ndarray
                contribution to total potential by this atom in 'box'
        """

        n_rad = basis['n']
        n_l = basis['l']
        r_o = basis['r_o']

        R, Theta, Phi = box['radial']
        Xm, Ym, Zm = box['mesh']

        # Automatically detect whether entire charge density or only surrounding
        # box was provided

        timer.start('build:basis_functions', False)
        timer.start('build:basis_functions:angular', False)
        #Build angular part of basis functions
        if not isinstance(angs, list):
            angs = []
            for l in range(n_l):
                angs.append([])
                ang_l = self.angulars_real(l, Theta, Phi)
                for m in range(-l, l + 1):
                    # angs[l].append(sph_harm(m, l, Phi, Theta).conj()) TODO: In theory should be conj!?
                    # angs[l].append(self.angulars(l, m, Theta, Phi))
                    angs[l].append(ang_l[l + m])

        timer.stop('build:basis_functions:angular', False)
        timer.start('build:basis_functions:radial', False)
        #Build radial part of b.f.
        if not isinstance(W, np.ndarray):
            W = self.get_W(basis)  # Matrix to orthogonalize radial basis

        rads = self.radials(R, basis, W)

        timer.stop('build:basis_functions:radial', False)
        timer.start('build:build', False)
        if  config.OptLevel > 1:
            v = np.zeros_like(Xm, dtype=np.float64)
            idx = 0

            timer.stop('build:basis_functions', False)
            for n in range(n_rad):
                for l in range(n_l):
                    for m in range(2 * l + 1):
                        v += coeffs[idx] * angs[l][m] * rads[n]
                        idx += 1
        else:
            coeffs_rs = coeffs.reshape(n_rad,n_l**2)
            angs_flat = np.array([a for ang in angs for a in ang]).reshape(n_l**2, *Xm.shape)
            rads = np.array(rads)

            v = np.einsum('nl,lijk,nijk-> ijk',coeffs_rs,angs_flat,rads)
        timer.stop('build:build', False)
        return v

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
        if rho.shape == Xm.shape:
            small_rho = True
        else:
            small_rho = False

        #Build angular part of basis functions
        if not isinstance(angs, list):
            angs = []
            for l in range(n_l):
                angs.append([])
                ang_l = self.angulars_real(l, Theta, Phi)
                for m in range(-l, l + 1):
                    angs[l].append(ang_l[l + m])

        #Build radial part of b.f.
        if not isinstance(W, np.ndarray):
            W = self.get_W(basis)  # Matrix to orthogonalize radial basis

        rads = self.radials(R, basis, W)

        timer.start('project:project', False)

        if not small_rho:
            srho = rho[Xm, Ym, Zm]

        #zero_pad_angs (so that it can be converted to numpy array):
        zeropad = np.zeros_like(Xm, dtype=np.float64)
        angs_padded = []
        for l in range(n_l):
            angs_padded.append([zeropad] * (n_l - l) + angs[l] + [zeropad] * (n_l - l))
        angs_padded = np.array(angs_padded)

        rads = np.array(rads) * self.V_cell
        coeff_array = np.einsum('lmijk,nijk,ijk -> nlm', angs_padded, rads, srho, optimize=config.OptLevel>0)
        coeff = []

        #remove zero padding from m
        for n in range(n_rad):
            for l in range(n_l):
                for m in range(2 * n_l + 1):
                    if abs(m - n_l) <= l:
                        coeff.append(coeff_array[n, l, m])

        timer.stop('project:project', False)
        if return_dict:
            return coeff_dict, angs
        else:
            return np.array(coeff).reshape(1, -1), angs

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
        if pos.shape != (1, 3) and (pos.ndim != 1 or len(pos) != 3):
            raise Exception('please provide only one point for pos. shape = {}'.format(pos.shape))

        pos = pos.flatten()

        #Create box with max. distance = radius
        rmax = np.ceil(radius / self.a).astype(int).tolist()
        Xm, Ym, Zm = mesh_3d(self.U, self.a, scaled=False, rmax=rmax, indexing='ij')
        X, Y, Z = mesh_3d(self.U, self.a, scaled=True, rmax=rmax, indexing='ij')

        #Find mesh pos.
        cm = np.round(self.U_inv.dot(pos)).astype(int)
        dr = pos - self.U.dot(cm)
        X -= dr[0]
        Y -= dr[1]
        Z -= dr[2]

        Xm = (Xm + cm[0]) % self.grid[0]
        Ym = (Ym + cm[1]) % self.grid[1]
        Zm = (Zm + cm[2]) % self.grid[2]

        R = np.sqrt(X**2 + Y**2 + Z**2)

        Phi = np.arctan2(Y, X)
        Theta = np.arccos(Z / R, where=(R > 1e-15))
        Theta[R < 1e-15] = 0

        return {'mesh': [Xm, Ym, Zm], 'real': [X, Y, Z], 'radial': [R, Theta, Phi]}


class OrthoProjector(DefaultProjector):

    _registry_name = 'ortho'

    @classmethod
    def dg(cls, r, basis, a):
        r_o = basis['r_o']
        return cls.dg_compiled(r, r_o, a)

    @staticmethod
    @jit(nopython=True)
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
        N = np.sqrt(720*r_o**(11+2*a)*1/((2*a+11)*(2*a+10)*(2*a+9)*(2*a+8)*(2*a+7)*\
                                       (2*a+6)*(2*a+5)))
        return r * (r_o - r)**(a + 1) * (2 * r_o - (a + 4) * r) / N

    @classmethod
    def g(cls, r, basis, a):
        r_o = basis['r_o']
        return cls.g_compiled(r, r_o, a)

    @staticmethod
    @jit(nopython=True)
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
        N = np.sqrt(720*r_o**(11+2*a)*1/((2*a+11)*(2*a+10)*(2*a+9)*(2*a+8)*(2*a+7)*\
                                           (2*a+6)*(2*a+5)))
        return (r)**(2) * (r_o - r)**(a + 2) / N

    @staticmethod
    def orthogonalize(func, r, basis, W):
        r_o = basis['r_o']
        result = np.zeros([len(W)] + list(r.shape))
        for k in range(0, len(W)):
            rad = func(r, basis, k + 1)
            for j in range(0, len(W)):
                result[j] += W[j, k] * rad
        result[:, r > r_o] = 0
        return result

    @classmethod
    def dradials(cls, r, basis, W):
        '''
        Get derivative of orthonormal radial basis functions

        Parameters
        -------
            r: float
                radius
            r_o: float
                outer radial cutoff
            W: np.ndarray
                orthogonalization matrix
        Returns
        -------
            np.ndarray
                radial functions
        '''
        return cls.orthogonalize(cls.dg, r, basis, W)

    @classmethod
    def radials(cls, r, basis, W):
        '''
        Get orthonormal radial basis functions

        Parameters
        -------

            r: float
                radius
            r_o: float
                outer radial cutoff
            W: np.ndarray
                orthogonalization matrix

        Returns
        -------

            np.ndarray
                radial functions
        '''
        return cls.orthogonalize(cls.g, r, basis, W)

    @classmethod
    def get_W(cls, basis):
        '''
        Get matrix to orthonormalize radial basis functions

        Parameters
        -------

            r_o: float
                outer radial cutoff
            n: int
                max. number of radial functions

        Returns
        -------
            np.ndarray
                W, orthogonalization matrix
        '''
        return scipy.linalg.sqrtm(np.linalg.pinv(cls.S(basis)))

    @classmethod
    # @jit
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
                S_matrix[i, j] = np.sum(cls.g(r_grid, basis, i + 1) * cls.g(r_grid, basis, j + 1) * r_grid**2) * dr
        for i in range(nmax):
            for j in range(i + 1, nmax):
                S_matrix[j, i] = S_matrix[i, j]
        return S_matrix


class NonOrthoProjector(OrthoProjector):
    _registry_name = 'non-ortho'

    @staticmethod
    def dg(r, basis, a):
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
        r_o = basis['r_o']

        def dg_(r, r_o, a):
            return -(2 + a) * (r_o - r)**(a + 1)

        N = np.sqrt(r_o**(2 * a + 5 / (2 * a + 5)))
        return dg_(r, r_o, a) / N

    @staticmethod
    def g(r, basis, a):
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

        r_o = basis['r_o']

        def g_(r, r_o, a):
            return (r_o - r)**(a + 2)

        # Write out factorial fraction to avoid overflow
        N = np.sqrt(r_o**(2 * a + 5 / (2 * a + 5)))
        return g_(r, r_o, a) / N

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

        nmax = basis['n']
        S_matrix = np.zeros([nmax, nmax])
        for i in range(1, nmax + 1):
            for j in range(i, nmax + 1):
                S_matrix[i - 1, j - 1] = np.sqrt((5 + 2 * i) * (5 + 2 * j)) / (5 + i + j)
        for i in range(nmax):
            for j in range(i + 1, nmax):
                S_matrix[j, i] = S_matrix[i, j]
        return S_matrix


class BehlerProjector(OrthoProjector):

    _registry_name = 'behler'

    @staticmethod
    def g(r, basis, a):
        r_o = basis['r_o']
        sigma = basis.get('sigma', 0.005)
        mu = a / (basis['n'] + 1) * r_o
        return r * (r_o - r) * np.exp(-(r - mu)**2 / (sigma * r_o))

    @classmethod
    def get_W(cls, basis):
        return np.eye(basis['n'])


class DeltaProjector():
    def __init__(self, projector):
        """ Wrapper class that can store a constant basis set representation
        and subtract it from given densities (e.g. subtract contribution from
        core densities)
        """
        self.projector = projector
        self.constant_basis_rep = {}

    def set_constant_density(self, rho, positions, species):
        self.constant_rho = np.array(rho)
        print('NeuralXC: set_constant_density called ')
        self.constant_basis_rep = \
            self.projector.get_basis_rep(rho, positions, species)
        self.positions = positions
        self.species = species

    def get_basis_rep(self, rho, positions, species, **kwargs):
        basis_rep = self.projector.get_basis_rep(rho, positions, species)
        if positions.shape != self.positions.shape:
            index = np.where(np.all(self.positions[species[0] == np.array(self.species)] == positions[0], axis=-1))
            basis_rep[species[0]] -= self.constant_basis_rep[species[0]][index]
        else:
            for spec in basis_rep:
                basis_rep[spec] -= self.constant_basis_rep[spec]
        return basis_rep

    def get_V(self, dEdC, positions, species, calc_forces=False, rho=None, **kwargs):

        if isinstance(rho, np.ndarray):
            rho = rho - self.constant_rho

        return self.projector.get_V(dEdC, positions, species, calc_forces, rho)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.projector, attr)


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
    x_pbc = list(range(0, rmax[0] + 1)) + list(range(-rmax[0], 0))
    y_pbc = list(range(0, rmax[1] + 1)) + list(range(-rmax[1], 0))
    z_pbc = list(range(0, rmax[2] + 1)) + list(range(-rmax[2], 0))

    Xm, Ym, Zm = np.meshgrid(x_pbc, y_pbc, z_pbc, indexing=indexing)

    Rm = np.concatenate([Xm.reshape(*Xm.shape, 1), Ym.reshape(*Xm.shape, 1), Zm.reshape(*Xm.shape, 1)], axis=3)

    if scaled:
        R = np.einsum('ij,klmj -> iklm', U, Rm)
        X = R[0, :, :, :]
        Y = R[1, :, :, :]
        Z = R[2, :, :, :]
        return X, Y, Z
    else:
        return Xm, Ym, Zm


def M_make_complex(n_l):
    """Get a matrix to convert real into complex tensors

    Parameters
    -------

        n_l: int,
            maximum angular momentum

    Returns
    -------

        M : np.ndarray,
            conversion matrix
    """
    M = np.zeros([n_l**2, n_l**2], dtype=complex)
    tensor = {}
    cnt = 0
    for l in range(n_l):
        for m in range(-l, l + 1):
            tensor['{},{}'.format(l, m)] = cnt
            cnt += 1

    idx = 0
    for l in range(n_l):
        for m in range(-l, 0):
            M[idx, tensor['{},{}'.format(l, -m)]] = (-1)**m * 1 / np.sqrt(2)
            M[idx, tensor['{},{}'.format(l, m)]] = -(-1)**m * 1j / np.sqrt(2)
            idx += 1
        M[idx, tensor['{},{}'.format(l, 0)]] = 1
        idx += 1
        for m in range(1, l + 1):
            M[idx, tensor['{},{}'.format(l, m)]] = 1 / np.sqrt(2)
            M[idx, tensor['{},{}'.format(l, -m)]] = 1j / np.sqrt(2)
            idx += 1
    return M

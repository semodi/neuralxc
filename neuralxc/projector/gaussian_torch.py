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
try:
    import torch
    TorchModule = torch.nn.Module
except ModuleNotFoundError:
    class TorchModule:
         def __init__(self):
             pass
from .projector import DefaultProjector
from .projector_torch import DefaultProjectorTorch
from .projector_gaussian import parse_basis
import neuralxc

GAMMA = torch.from_numpy(np.array([1/2,3/4,15/8,105/16,945/32,10395/64,135135/128])*np.sqrt(np.pi))

class GaussianProjector(DefaultProjectorTorch):

    _registry_name = 'gaussian_torch'
    _unit_test = False

    def __init__(self, unitcell, grid, basis_instructions, **kwargs):
        TorchModule.__init__(self)
        full_basis, basis_strings = parse_basis(basis_instructions)
        basis = {key:val for key,val in basis_instructions.items()}
        basis.update(full_basis)
        self.basis_strings = basis_strings
        DefaultProjectorTorch.__init__(self, unitcell, grid, basis, **kwargs)


    def forward_basis(self, positions, unitcell, grid, my_box):

        r_o_max = np.max([np.max(b['r_o']) for b in self.basis[self.species]])

        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        box, mesh = self.box_around(positions, r_o_max, my_box)
        box['mesh'] = mesh
        rad, ang  =  self.get_basis_on_mesh(box, basis, self.W[self.species])
        return rad, ang, torch.cat([mesh, box['radial']])

    def forward_fast(self, rho, positions, unitcell, grid, radials, angulars, my_box):
        self.set_cell_parameters(unitcell, grid)
        basis = self.basis[self.species]
        box = {}
        box['mesh'] = my_box[:3]
        box['radial'] = my_box[3:]
        Xm, Ym, Zm = box['mesh'].long()
        return  self.project_onto(rho[Xm,Ym,Zm], radials, angulars, basis, self.basis_strings[self.species], box)

    def get_basis_on_mesh(self, box, basis_instructions, W):

        angs = []
        rads = []

        box['radial'] = torch.stack(box['radial'])
        for ib, basis in enumerate(basis_instructions):
            l = basis['l']
            r_o_max = np.max(basis['r_o'])
            # filt = (box['radial'][0] <= r_o_max)
            filt = (box['radial'][0] <= 1000000)
            box_rad = box['radial'][:,filt]
            box_m = box['mesh'][:,filt]
            ang = torch.zeros([2*l+1,filt.size()[0]], dtype=torch.double)
            rad = torch.zeros([len(basis['r_o']),filt.size()[0]], dtype=torch.double)
            # ang[:,filt] = torch.stack(self.angulars_real(l, box_rad[1], box_rad[2])) # shape (m, x, y, z)
            # rad[:,filt] = torch.stack(self.radials(box_rad[0], [basis])[0]) # shape (n, x, y, z)
            # rads.append(rad)
            # angs.append(ang)
            angs.append(torch.stack(self.angulars_real(l, box_rad[1], box_rad[2]))) # shape (m, x, y, z)
            rads.append(torch.stack(self.radials(box_rad[0], [basis])[0])) # shape (n, x, y, z)

        return torch.cat(rads), torch.cat(angs)

    def project_onto(self, rho, rads, angs, basis_instructions, basis_string, box):

        rad_cnt = 0
        ang_cnt = 0
        coeff = []
        for basis in basis_instructions:
            # print(basis)
            l = basis['l']
            len_rad = len(basis['r_o'])
            rad = rads[rad_cnt:rad_cnt+len_rad]
            ang = angs[ang_cnt:ang_cnt + (2*l+1)]
            rad_cnt += len_rad
            ang_cnt += 2*l + 1
            r_o_max = np.max(basis['r_o'])
            filt = (box['radial'][0] <= r_o_max)
            # filt = (box['radial'][0] <= 1000000)
            rad *= self.V_cell
            # coeff.append(torch.einsum('i,mi,ni -> nm', rho[filt], ang[:,filt], rad[:,filt]).reshape(-1))
            coeff.append(torch.einsum('i,mi,ni -> nm', rho, ang, rad).reshape(-1))

        mol = gto.M(atom='O 0 0 0',
                    basis={'O': gtobasis.parse(basis_string)})
        bp = neuralxc.pyscf.BasisPadder(mol)

        coeff = torch.cat(coeff)

        sym = 'O'
        print(bp.indexing_l[sym][0])
        print(bp.indexing_r[sym][0])
        indexing_r = torch.from_numpy(np.array(bp.indexing_r[sym][0])).long()
        indexing_l = torch.from_numpy(np.array(bp.indexing_l[sym][0])).bool()
        coeff = coeff[indexing_r]

        coeff_out = torch.zeros([bp.max_n[sym] * (bp.max_l[sym] + 1)**2], dtype= torch.double)

        coeff_out[indexing_l] = coeff
        return coeff_out


    @classmethod
    def g(cls, r, r_o, alpha, l):
        fc = 1-(.5*(1-torch.cos(np.pi*r/r_o[0])))**8
        N = (2*alpha[0])**(l/2+3/4)*np.sqrt(2)/np.sqrt(GAMMA[l])
        f = r**l*torch.exp(-alpha[0]*r**2)*fc*N
        f[r>r_o[0]] = 0
        return f

    @classmethod
    def get_W(cls, basis):
        return np.eye(3)

    @classmethod
    def radials(cls, r, basis, W = None):
        result = []
        if isinstance(basis, list):
            for b in basis:
                res = []
                for ib, alpha in enumerate(b['alpha']):
                    res.append(cls.g(r, b['r_o'][ib], b['alpha'][ib], b['l']))
                result.append(res)
        elif isinstance(basis, dict):
                result.append([cls.g(r, basis['r_o'], basis['alpha'], basis['l'])])
        return result

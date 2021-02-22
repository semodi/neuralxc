"""
pyscf.py
Contains all routines concerned with PySCF.
For PySCF, projection integrals are computed analytical in the GTO basis and
no grid operations are necessary.
BasisPadder translates between NeuralXC and PySCF internal basis set orderings.
"""

import neuralxc
from abc import ABC, abstractmethod
import numpy as np
import pyscf
from pyscf import gto, dft
from pyscf.dft import RKS
from pyscf.scf.chkfile import load_scf
from functools import reduce
import math
from ..doc_inherit import doc_inherit
from ..base import ABCRegistry
from ..projector import BaseProjector
import os
from opt_einsum import contract

l_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'j': 7}
l_dict_inv = {l_dict[key]: key for key in l_dict}


def get_eri3c(mol, auxmol, op):
    """ Returns three center-one electron intergrals need for basis
    set projection.

    TODO: Name misleading as no electron repulsion integrals computed,
     will be changed in future versions.
    """
    pmol = mol + auxmol
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    if op == 'rij':
        eri3c = pmol.intor('int3c2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas))
    elif op == 'delta':
        eri3c = pmol.intor('int3c1e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas))
    else:
        raise ValueError('Operator {} not implemented'.format(op))

    return eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)


def get_coeff(dm, eri3c):
    """ Given a density matrix, return coefficients from basis set projection
    """
    return contract('ijk, ij -> k', eri3c, dm)


class PySCFProjector(BaseProjector):

    _registry_name = 'pyscf'

    def __init__(self, mol, basis_instructions, **kwargs):
        self.basis = basis_instructions
        self.initialize(mol)

    def initialize(self, mol, **kwargs):
        self.spec_agnostic = self.basis.get('spec_agnostic', False)
        self.op = self.basis.get('operator', 'delta').lower()
        self.delta = self.basis.get('delta', False)

        if self.delta:
            mf = RKS(mol)
            self.dm_init = mf.init_guess_by_atom()

        if self.spec_agnostic:
            basis = {}
            for atom_idx, _ in enumerate(mol.atom_charges()):
                sym = mol.atom_pure_symbol(atom_idx)
                if os.path.isfile(self.basis['basis']):
                    basis[sym] = gto.basis.parse(open(self.basis['basis'], 'r').read())
                else:
                    basis[sym] = gto.basis.load(self.basis['basis'], 'O')
        else:
            basis = self.basis['basis']

        auxmol = gto.M(atom=mol.atom, basis=basis)
        self.bp = BasisPadder(auxmol)
        self.eri3c = get_eri3c(mol, auxmol, self.op)
        self.mol = mol
        self.auxmol = auxmol

    def get_basis_rep(self, dm, **kwargs):
        # if not mol is None and mol.atom != self.mol.atom:
        #     self.initialize(mol)
        if self.delta:
            dm = dm - self.dm_init
        coeff = get_coeff(dm, self.eri3c)
        coeff = self.bp.pad_basis(coeff)

        if self.spec_agnostic:
            self.spec_partition = {sym: len(coeff[sym]) for sym in coeff}
            coeff_agn = np.concatenate([coeff[sym] for sym in coeff], axis=0)
            coeff = {'X': coeff_agn}

        return coeff

    def get_V(self, dEdC, **kwargs):
        if self.spec_agnostic:
            running_idx = 0
            for sym in self.spec_partition:
                dEdC[sym] = dEdC['X'][:, running_idx:running_idx + self.spec_partition[sym]]
                running_idx += self.spec_partition[sym]

            dEdC.pop('X')
        dEdC = self.bp.unpad_basis(dEdC)
        V = contract('ijk, k', self.eri3c, dEdC)
        return V


class BasisPadder():
    def __init__(self, mol):

        self.mol = mol

        max_l = {}
        max_n = {}
        sym_cnt = {}
        sym_idx = {}
        # Find maximum angular momentum and n for each species
        for atom_idx, _ in enumerate(mol.atom_charges()):
            sym = mol.atom_pure_symbol(atom_idx)
            if not sym in sym_cnt:
                sym_cnt[sym] = 0
                sym_idx[sym] = []
            sym_idx[sym].append(atom_idx)
            sym_cnt[sym] += 1

        for ao_idx, label in enumerate(mol.ao_labels(fmt=False)):
            sym = label[1]
            if not sym in max_l:
                max_l[sym] = 0
                max_n[sym] = 0

            n = int(label[2][:-1])
            max_n[sym] = max(n, max_n[sym])

            l = l_dict[label[2][-1]]
            max_l[sym] = max(l, max_l[sym])

        indexing_left = {sym: [] for sym in max_n}
        indexing_right = {sym: [] for sym in max_n}
        labels = mol.ao_labels()
        for sym in max_n:
            for idx in sym_idx[sym]:
                indexing_left[sym].append([])
                indexing_right[sym].append([])
                for n in range(1, max_n[sym] + 1):
                    for l in range(max_l[sym] + 1):
                        if any(['{} {} {}{}'.format(idx, sym, n, l_dict_inv[l]) in lab for lab in labels]):
                            indexing_left[sym][-1] += [True] * (2 * l + 1)
                            sidx = np.where(['{} {} {}{}'.format(idx, sym, n, l_dict_inv[l]) in lab
                                             for lab in labels])[0][0]
                            indexing_right[sym][-1] += np.arange(sidx, sidx + (2 * l + 1)).astype(int).tolist()
                        else:
                            indexing_left[sym][-1] += [False] * (2 * l + 1)

        self.sym_cnt = sym_cnt
        self.max_l = max_l
        self.max_n = max_n
        self.indexing_l = indexing_left
        self.indexing_r = indexing_right

    def get_basis_json(self):

        basis = {}

        for sym in self.sym_cnt:
            basis[sym] = {'n': self.max_n[sym], 'l': self.max_l[sym] + 1}

        if 'O' in basis:
            basis['X'] = {'n': self.max_n['O'], 'l': self.max_l['O'] + 1}

        return basis

    def pad_basis(self, coeff):
        # Mimu = None
        coeff_out = {
            sym: np.zeros([self.sym_cnt[sym], self.max_n[sym] * (self.max_l[sym] + 1)**2])
            for sym in self.indexing_l
        }

        cnt = {sym: 0 for sym in self.indexing_l}

        for aidx, slice in enumerate(self.mol.aoslice_by_atom()):
            sym = self.mol.atom_pure_symbol(aidx)
            coeff_out[sym][cnt[sym], self.indexing_l[sym][cnt[sym]]] = coeff[slice[-2]:slice[-1]][
                np.array(self.indexing_r[sym][cnt[sym]]) - slice[-2]]
            cnt[sym] += 1

        return coeff_out

    def unpad_basis(self, coeff):

        cnt = {sym: 0 for sym in self.indexing_l}
        coeff_out = np.zeros(len(self.mol.ao_labels()))
        for aidx, slice in enumerate(self.mol.aoslice_by_atom()):
            sym = self.mol.atom_pure_symbol(aidx)
            coeff_in = coeff[sym]
            if coeff_in.ndim == 3: coeff_in = coeff_in[0]
            coeff_out[slice[-2]:slice[-1]][np.array(self.indexing_r[sym][cnt[sym]]) -
                                           slice[-2]] = coeff_in[cnt[sym], self.indexing_l[sym][cnt[sym]]]
            cnt[sym] += 1

        return coeff_out

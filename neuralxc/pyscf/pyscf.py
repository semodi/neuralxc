"""
pyscf.py

Defines the modified effective potential veff_mod that allows the integration
of NeuralXC models in PySCF calculations. Provides utility functions for NeuralXC
PySCF interoperability.
"""
import neuralxc
from abc import ABC, abstractmethod
import pyscf
from pyscf import gto, dft
from pyscf.dft import RKS
from pyscf.scf import hf, RHF, RKS
from pyscf.scf.chkfile import load_scf
from pyscf.lib.numpy_helper import NPArrayWithTag
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
from ..projector import BaseProjector
import neuralxc
import os
from pylibnxc import get_nxc_adapter
from glob import glob

LAMBDA = 0.1

l_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'j': 7}
l_dict_inv = {l_dict[key]: key for key in l_dict}


def RKS(mol, nxc='', **kwargs):
    """ Wrapper for the pyscf RKS (restricted Kohn-Sham) class
    that uses a NeuralXC potential
    """
    mf = dft.RKS(mol, **kwargs)
    if not nxc is '':
        model = neuralxc.PySCFNXC(nxc)
        model.initialize(mol)
        mf.get_veff = veff_mod(mf, model)
    return mf


def compute_KS(atoms, path='pyscf.chkpt', basis='ccpvdz', xc='PBE', nxc='', **kwargs):
    """ Given an ase atoms object, run a pyscf RKS calculation on it and
    return the results
    """
    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]
    mol = gto.M(atom=mol_input, basis=basis, **kwargs)
    if nxc:
        model_paths = glob(nxc + '/*')
        if any(['projector' in path for path in model_paths]):
            mf = get_nxc_adapter('pyscf', nxc)  # Model that uses projector on radial grid
        else:
            mf = RKS(mol, nxc=nxc)  # Model that uses overlap integrals and density matrix
    else:
        mf = dft.RKS(mol)
    mf.set(chkfile=path)
    mf.xc = xc
    mf.kernel()
    return mf, mol


def veff_mod(mf, model):
    """ Wrapper to get the modified get_veff() that uses a NeuralXC
    potential
    """
    def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        veff = pyscf.dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        vnxc = NPArrayWithTag(veff.shape)
        nxc = model.get_V(dm)
        vnxc[:, :] = nxc[1][:, :]
        vnxc.exc = nxc[0]
        vnxc.ecoul = 0
        veff[:, :] += vnxc[:, :]
        veff.exc += vnxc.exc
        return veff

    return get_veff

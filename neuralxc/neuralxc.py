"""
neuralxc.py
Implementation of a machine learned density functional
Interfaces to pyblibnxc classes. Here for compatibility reasons
"""
import json
import os
from glob import glob

from pylibnxc import AtomicFunc
from pylibnxc.adapters import Hartree

from neuralxc.projector import DensityProjector
from neuralxc.utils import ConfigFile


class PySCFNXC(AtomicFunc):
    def __init__(self, path):
        model_paths = glob(path + '/*')
        for mp in model_paths:
            if 'bas.json' == os.path.basename(mp):
                self.basis = ConfigFile(mp)['preprocessor']

        super().__init__(path)

    def initialize(self, mol):
        self.projector = DensityProjector(basis_instructions=self.basis, mol=mol)
        self.projector.initialize(mol)

    def get_V(self, dm):
        C = self.projector.get_basis_rep(dm)
        output = self.compute({'c': C}, do_forces=False, edens=False)
        E = output['zk']
        dEdC = output['dEdC']
        V = self.projector.get_V(dEdC)
        E /= Hartree
        V /= Hartree
        return E, V


class NeuralXC(AtomicFunc):
    def get_V(self, rho, calc_forces=False):
        output = self.compute({'rho': rho}, do_forces=calc_forces, edens=False)
        E, V = output['zk'], output['vrho']
        if calc_forces:
            forces = output['forces']
            V = (V, forces)
        return E, V

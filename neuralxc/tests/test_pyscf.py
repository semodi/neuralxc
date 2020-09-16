import neuralxc as xc
import pytest
import sys
import numpy as np
import os
from neuralxc.doc_inherit import doc_inherit
from abc import ABC, abstractmethod
import dill as pickle
import copy
import matplotlib.pyplot as plt
from neuralxc.constants import Bohr, Hartree
from neuralxc.drivers import *
from neuralxc.engines import Engine
import shutil
try:
    import pyscf
    pyscf_found = True
except ModuleNotFoundError:
    pyscf_found = False

test_dir = os.path.dirname(os.path.abspath(__file__))

if 'driver_data_tmp' in os.listdir(test_dir):
    shutil.rmtree(test_dir + '/driver_data_tmp')


def shcopytree(src, dest):
    try:
        shutil.copytree(src, dest)
    except FileExistsError:
        shutil.rmtree(dest)
        shutil.copytree(src, dest)


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
def test_radial_model():
    from pyscf import gto, dft
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = dft.RKS(mol)
    mf.xc = 'PBE'
    mf.grids.level = 5
    mf.kernel()

    model = xc.NeuralXCJIT(test_dir[:-len('neuralxc/tests/')] + '/examples/models/NXC-W01/nxc_w01_radial.jit')
    rho = pyscf.dft.numint.get_rho(mf._numint, mol, mf.make_rdm1(), mf.grids)

    model.initialize(unitcell = mf.grids.coords, grid = mf.grids.weights,
                     positions = np.array([[0,0,0],[0,1,0],[0,0,1]])/Bohr,
                     species = ['O','H','H'])

    res = model.get_V(rho)[0]
    assert np.allclose(res,np.load(test_dir + '/rad_energy.npy'))

@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.pyscf
def test_adiabatic():
    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    fetch_default_driver(kind='pre', hint='./pre_hint.json')
    adiabatic_driver('benzene_small.traj', 'pre.json', 'hyper.json', maxit=2)
    os.chdir(test_dir + '/driver_data_tmp')
    engine = Engine('pyscf', nxc='it0/best_model')
    engine.compute(read('benzene_small.traj', '0'))

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

if __name__ == '__main__':
    test_iterative()

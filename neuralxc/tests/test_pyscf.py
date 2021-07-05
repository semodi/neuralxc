import copy
import os
import shutil
import sys
from abc import ABC, abstractmethod

import dill as pickle
import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase.io import read

import neuralxc as xc
from neuralxc.constants import Bohr, Hartree
from neuralxc.drivers import *
from neuralxc.engines import Engine
from neuralxc.utils import ConfigFile

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
    from pyscf import dft, gto
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = dft.RKS(mol)
    mf.xc = 'PBE'
    mf.grids.level = 5
    mf.kernel()

    model = xc.NeuralXC(test_dir[:-len('neuralxc/tests/')] + '/examples/models/NXC-W01/nxc_w01_radial.jit')
    rho = pyscf.dft.numint.get_rho(mf._numint, mol, mf.make_rdm1(), mf.grids)

    model.initialize(grid_coords=mf.grids.coords,
                     grid_weights=mf.grids.weights,
                     positions=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) / Bohr,
                     species=['O', 'H', 'H'])

    res = model.get_V(rho)[0]
    assert np.allclose(res, np.load(test_dir + '/rad_energy.npy'))


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.pyscf
def test_pre():
    try:
        shutil.rmtree(test_dir + '/driver_data_tmp')
    except:
        pass
    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    run_engine_driver('benzene_small.traj', 'pre_rad.json', workdir='workdir_engine')

    pre_driver('benzene_small.traj', 'workdir_engine', 'pre_rad.json', 'data.hdf5/test/test')
    pre = ConfigFile('pre_rad.json')
    pre['preprocessor']['grad'] = 1
    open('pre_rad.json', 'w').write(json.dumps(pre.__dict__))
    pre_driver('benzene_small.traj', 'workdir_engine', 'pre_rad.json', 'data.hdf5/test/test1')

    pre = ConfigFile('pre_rad.json')
    pre['preprocessor']['grad'] = 2
    open('pre_rad.json', 'w').write(json.dumps(pre.__dict__))
    pre_driver('benzene_small.traj', 'workdir_engine', 'pre_rad.json', 'data.hdf5/test/test2')

    with h5py.File('data.hdf5', 'r') as f:
        for hashkey in f['/test/test/density']:
            data0 = f['/test/test/density/' + hashkey][:]
        for hashkey in f['/test/test1/density']:
            data1 = f['/test/test1/density/' + hashkey][:]
        for hashkey in f['/test/test2/density']:
            data2 = f['/test/test2/density/' + hashkey][:]

    assert data0.shape[-1] * 2 == data1.shape[-1]
    assert data0.shape[-1] * 4 == data2.shape[-1]
    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.pyscf
@pytest.mark.parametrize('projector', ['ga_ana','ga_rad','or_rad', 'ga_ana_f','ga_rad_f'])
def test_sc(projector):
    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')
    sc_driver('water.traj', 'pre_sc_{}.json'.format(projector),
        'hyper.json', maxit=1, hyperopt=True)
    os.chdir(test_dir + '/driver_data_tmp')

    # engine = Engine('pyscf', nxc='testing/nxc.jit', basis='sto3g')
    # engine.compute(read('testing.traj', '0'))

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


def test_pyscf_radial():
    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    serialize('model', 'benzene.pyscf.jit', as_radial=False)
    engine = Engine('pyscf', nxc='benzene.pyscf.jit')
    atoms = engine.compute(read('benzene_small.traj', '0'))
    serialize('model', 'benzene.pyscf_radial.jit', as_radial=True)
    engine = Engine('pyscf', nxc='benzene.pyscf_radial.jit')
    atoms_rad = engine.compute(read('benzene_small.traj', '0'))

    assert np.allclose(atoms_rad.get_potential_energy(), atoms.get_potential_energy())

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')
    pass


if __name__ == '__main__':
    test_adiabatic()

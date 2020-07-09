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


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.pyscf
@pytest.mark.pyscf_iterative
def test_iterative():
    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    fetch_default_driver(kind='pre', hint='./pre_hint.json')
    workflow_driver('benzene_small.traj', 'pre.json', 'hyper.json', maxit=2, stop_early=False)

    engine = Engine('pyscf', nxc='it1/nxc')
    engine.compute(read('benzene_small.traj', '0'))

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


if __name__ == '__main__':
    test_iterative()

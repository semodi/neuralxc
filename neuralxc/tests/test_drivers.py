import neuralxc as xc
import pytest
import sys
import numpy as np
import os
from neuralxc.doc_inherit import doc_inherit
from abc import ABC, abstractmethod
import pickle
import copy
import matplotlib.pyplot as plt
from neuralxc.constants import Bohr, Hartree
from neuralxc.drivers import *
import shutil
try:
    import ase
    ase_found = True
except ModuleNotFoundError:
    ase_found = False

test_dir = os.path.dirname(os.path.abspath(__file__))

if 'driver_data_tmp' in os.listdir(test_dir):
    shutil.rmtree(test_dir + '/driver_data_tmp')


def shcopytree(src, dest):
    try:
        shutil.copytree(src, dest)
    except FileExistsError:
        shutil.rmtree(dest)
        shutil.copytree(src, dest)


@pytest.mark.driver
@pytest.mark.driver_fit
def test_fit():
    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')
    fit_driver(preprocessor='pre.json', config='hyper.json', sets='sets.inp', hyperopt=True)

    fit_driver(preprocessor='pre.json', config='hyper.json', model='model', ensemble=True, sets='sets.inp')
    fit_driver(preprocessor='pre.json', config='hyper.json', model='best_model', ensemble=True, sets='sets.inp')
    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_eval
def test_eval():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    eval_driver(hdf5=['data.hdf5', 'system/it1', 'system/ref'])

    eval_driver(model='model', hdf5=['data.hdf5', 'system/it0', 'system/ref'])

    eval_driver(model='model', hdf5=['data.hdf5', 'system/it0', 'system/ref'], predict=True, dest='prediction')

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_convert
def test_convert():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    fit_driver(preprocessor='pre.json', config='hyper.json', sets='sets.inp', hyperopt=True)
    convert_tf(tf_path='best_model', np_path='converted')

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_chain
def test_chain_merge():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    chain_driver(config='hyper.json', model='model', dest='chained')
    fit_driver(preprocessor='pre.json', config='hyper.json', model='chained', sets='sets.inp', hyperopt=True)

    merge_driver(chained='best_model', merged='merged')

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_ensemble
@pytest.mark.parametrize('operation', ['sum', 'mean'])
@pytest.mark.parametrize('estonly', [False, True])
def test_ensemble(operation, estonly):

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    ensemble_driver(operation=operation, dest=operation, models=['model', 'model'], estonly=estonly)

    eval_driver(model='model', hdf5=['data.hdf5', 'system/it0', 'system/ref'], predict=True, dest='single_pred')

    eval_driver(model=operation, hdf5=['data.hdf5', 'system/it0', 'system/ref'], predict=True, dest='ensemble_pred')

    if operation == 'sum':
        assert np.allclose(np.load('single_pred.npy') * 2, np.load('ensemble_pred.npy'), atol=1e-8, rtol=1e-5)
    elif operation == 'mean':
        assert np.allclose(np.load('single_pred.npy'), np.load('ensemble_pred.npy'), atol=1e-8, rtol=1e-5)

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_data
def test_data():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    add_data_driver(hdf5='data.hdf5',
                    system='system',
                    method='test',
                    add=['energy', 'forces'],
                    traj='results.traj',
                    override=True,
                    zero=10)

    add_data_driver(hdf5='data.hdf5',
                    system='system',
                    method='test',
                    add=['energy', 'forces'],
                    traj='results.traj',
                    override=True,
                    zero=None)

    split_data_driver(hdf5='data.hdf5', group='system/it0', label='training', slice=':3', comp='testing')

    delete_data_driver(hdf5='data.hdf5', group='system/it0/testing')

    sample_driver(preprocessor='pre.json', size=5, dest='sample.npy', hdf5=['data.hdf5', 'system/it0'])
    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

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
from types import SimpleNamespace as SN
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
    fit_driver(
        SN(preprocessor='pre.json',
           config='hyper.json',
           mask=False,
           sample='',
           cutoff=0.0,
           model='',
           ensemble=False,
           sets='sets.inp',
           hyperopt=True))

    fit_driver(
        SN(preprocessor='pre.json',
           config='hyper.json',
           mask=False,
           sample='',
           cutoff=0.0,
           model='model',
           ensemble=True,
           sets='sets.inp',
           hyperopt=False))
    fit_driver(
        SN(preprocessor='pre.json',
           config='hyper.json',
           mask=False,
           sample='',
           cutoff=0.0,
           model='best_model',
           ensemble=True,
           sets='sets.inp',
           hyperopt=False))
    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

@pytest.mark.driver
@pytest.mark.driver_eval
def test_eval():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    eval_driver(SN(model = '',hdf5=['data.hdf5','system/it1',
            'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=False))

    eval_driver(SN(model = 'model',hdf5=['data.hdf5','system/it0',
            'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=False))

    eval_driver(SN(model = 'model',hdf5=['data.hdf5','system/it0',
            'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=True, dest='prediction'))

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

@pytest.mark.driver
@pytest.mark.driver_convert
def test_convert():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')


    fit_driver(
            SN(preprocessor='pre.json',
               config='hyper.json',
               mask=False,
               sample='',
               cutoff=0.0,
               model='',
               ensemble=False,
               sets='sets.inp',
               hyperopt=True))
    convert_tf(SN(tf='best_model', np='converted'))

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

@pytest.mark.driver
@pytest.mark.driver_chain
def test_chain_merge():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    chain_driver(SN(config='hyper.json', model='model', dest='chained'))
    fit_driver(
        SN(preprocessor='pre.json',
           config='hyper.json',
           mask=False,
           sample='',
           cutoff=0.0,
           model='chained',
           ensemble=False,
           sets='sets.inp',
           hyperopt=True))

    merge_driver(SN(chained='best_model', merged='merged'))

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

@pytest.mark.driver
@pytest.mark.driver_ensemble
@pytest.mark.parametrize('operation',['sum','mean'])
def test_ensemble(operation):

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    ensemble_driver(SN(operation=operation, dest=operation, models=['model', 'model']))

    eval_driver(SN(model = 'model', hdf5=['data.hdf5','system/it0',
            'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=True, dest='single_pred'))

    eval_driver(SN(model = operation,hdf5=['data.hdf5','system/it0',
            'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=True, dest='ensemble_pred'))

    if operation == 'sum':
        assert np.allclose(np.load('single_pred.npy')*2, np.load('ensemble_pred.npy'),atol = 1e-8, rtol = 1e-5)
    elif operation == 'mean':
        assert np.allclose(np.load('single_pred.npy'), np.load('ensemble_pred.npy'), atol = 1e-8, rtol = 1e-5)

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_data
def test_data():

    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')


    add_data_driver(
                    SN(hdf5='data.hdf5',
                       system='system',
                       method='test',
                       add=['energy','forces'],
                       traj='results.traj',
                       density='',
                       override=True,
                       slice=':',
                       zero=10))

    add_data_driver(
                    SN(hdf5='data.hdf5',
                       system='system',
                       method='test',
                       add=['energy','forces'],
                       traj='results.traj',
                       density='',
                       override=True,
                       slice=':',
                       zero=None))


    split_data_driver(SN(hdf5='data.hdf5',
                       group='system/it0',
                       label='training',
                       slice=':3',
                       comp='testing'))

    delete_data_driver(SN(hdf5='data.hdf5',
                       group='system/it0/testing'))

    sample_driver(SN(preprocessor='pre.json',
                        size= 5,
                        dest= 'sample.npy',
                        hdf5=['data.hdf5','system/it0'],
                        cutoff=0.0))
    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

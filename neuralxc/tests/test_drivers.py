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
import shutil

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
    # Fit model
    fit_driver(preprocessor='pre.json', hyper='hyper.json', sets='sets.inp', hyperopt=True)
    # Continue training
    fit_driver(preprocessor='pre.json', hyper='hyper.json', model='best_model', ensemble=False, sets='sets.inp')

    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_fit
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

def test_compile():


    os.chdir(test_dir)
    shcopytree(test_dir + '/driver_data', test_dir + '/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(test_dir + '/driver_data_tmp')

    compile('model','model.jit', False)
    os.chdir(cwd)
    shutil.rmtree(test_dir + '/driver_data_tmp')

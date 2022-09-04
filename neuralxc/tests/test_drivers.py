import copy
import os
import shutil
import sys
from abc import ABC, abstractmethod

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pytest

import neuralxc as xc
from neuralxc.constants import Bohr, Hartree
from neuralxc.drivers import *

test_dir = os.path.dirname(os.path.abspath(__file__))

if 'driver_data_tmp' in os.listdir(test_dir):
    shutil.rmtree(f'{test_dir}/driver_data_tmp')


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
    shcopytree(f'{test_dir}/driver_data', f'{test_dir}/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(f'{test_dir}/driver_data_tmp')
    # Fit model
    fit_driver(preprocessor='pre.json', hyper='hyper.json', sets='sets.inp', hyperopt=True)
    # Continue training
    fit_driver(preprocessor='pre.json', hyper='hyper.json', model='best_model', sets='sets.inp')

    os.chdir(cwd)
    shutil.rmtree(f'{test_dir}/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_fit
def test_eval():
    os.chdir(test_dir)
    shcopytree(f'{test_dir}/driver_data', f'{test_dir}/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(f'{test_dir}/driver_data_tmp')
    eval_driver(hdf5=['data.hdf5', 'system/it1', 'system/ref'])

    eval_driver(model='model_old', hdf5=['data.hdf5', 'system/it0', 'system/ref'])

    eval_driver(model='model_old', hdf5=['data.hdf5', 'system/it0', 'system/ref'], predict=True, dest='prediction')

    os.chdir(cwd)
    shutil.rmtree(f'{test_dir}/driver_data_tmp')


@pytest.mark.driver
@pytest.mark.driver_data
def test_data():

    os.chdir(test_dir)
    shcopytree(f'{test_dir}/driver_data', f'{test_dir}/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(f'{test_dir}/driver_data_tmp')

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
    shutil.rmtree(f'{test_dir}/driver_data_tmp')


def test_serialize():

    os.chdir(test_dir)
    shcopytree(f'{test_dir}/driver_data', f'{test_dir}/driver_data_tmp')
    cwd = os.getcwd()
    os.chdir(f'{test_dir}/driver_data_tmp')

    serialize('model', 'model.jit', False)
    os.chdir(cwd)
    shutil.rmtree(f'{test_dir}/driver_data_tmp')

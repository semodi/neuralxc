import json
import glob
import h5py
from ase.io import read
from neuralxc.symmetrizer import symmetrizer_factory
from neuralxc.formatter import atomic_shape, system_shape, SpeciesGrouper
from neuralxc.ml.transformer import GroupedPCA, GroupedVarianceThreshold
from neuralxc.ml.transformer import GroupedStandardScaler
from neuralxc.ml import NetworkEstimator as NetworkWrapper
from neuralxc.ml import NXCPipeline
from neuralxc.ml.ensemble import StackedEstimator, ChainedEstimator
from neuralxc.ml.network import load_pipeline, NumpyNetworkEstimator
from neuralxc.preprocessor import Preprocessor
from neuralxc.datastructures.hdf5 import *
from neuralxc.ml.utils import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import pandas as pd
from pprint import pprint
from dask.distributed import Client, LocalCluster
from sklearn.externals.joblib import parallel_backend
import time
import os
import shutil
from collections import namedtuple
import hashlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import neuralxc as xc
import sys
import copy
import pickle
from types import SimpleNamespace as SN
from .data import *
from .model import *
os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def plot_basis(args):
    """ Plots a set of basis functions specified in .json file"""

    basis_instructions = json.loads(open(args.basis, 'r').read())
    projector = xc.projector.DensityProjector(np.eye(3), np.ones(3), basis_instructions['basis'])

    for spec in basis_instructions['basis']:
        if not len(spec) == 1: continue
        basis = basis_instructions['basis'][spec]
        n = basis_instructions['basis'][spec]['n']
        W = projector.get_W(basis)
        r = np.linspace(0, basis['r_o'], 500)
        radials = projector.radials(r, basis, W)
        for rad in radials:
            plt.plot(r, rad)
        plt.show()


def parse_sets_input(path):
    """ Reads a file containing the sets used for fitting

    Parameters
    ----------
    path: str
        Path to file containing dataset names

    Returns
    --------
    hdf5, list
        hdf5[0] : datafile location
        hdf5[1],hdf5[2]: lists of baseline(,target) datasets
    """
    hdf5 = ['', [], []]
    with open(path, 'r') as setsfile:
        line = setsfile.readline().rstrip()
        hdf5[0] = line  #datafile location
        line = setsfile.readline().rstrip()
        while (line):
            split = line.split()
            hdf5[1].append(split[0])
            hdf5[2].append(split[1])
            line = setsfile.readline().rstrip()
    return hdf5


def pre_driver(args):
    """ Preprocess electron densities obtained from electronic structure
    calculations
    """
    preprocessor = args.preprocessor
    dest = args.dest
    xyz = args.xyz
    mask = args.mask

    if not mask:
        pre = json.loads(open(preprocessor, 'r').read())
    else:
        pre = {}

    if 'traj_path' in pre and pre['traj_path'] != '':
        atoms = read(pre['traj_path'], ':')
        trajectory_path = pre['traj_path']
    elif xyz != '':
        atoms = read(xyz, ':')
        trajectory_path = xyz
    else:
        raise ValueError('Must specify path to to xyz file')

    preprocessor = get_preprocessor(preprocessor, mask, xyz)
    if not mask:
        start = time.time()

        if 'hdf5' in dest:
            dest_split = dest.split('/')
            file, system, method = dest_split + [''] * (3 - len(dest_split))
            workdir = '.tmp'
            delete_workdir = True
        else:
            workdir = dest
            delete_workdir = False

        try:
            os.mkdir(workdir)
        except FileExistsError:
            delete_workdir = False
            pass
        print('======Projecting onto basis sets======')
        basis_grid = get_basis_grid(pre)['preprocessor__basis_instructions']

        for basis_instr in basis_grid:
            preprocessor.basis_instructions = basis_instr
            filename = os.path.join(workdir, basis_to_hash(basis_instr) + '.npy')
            data = preprocessor.fit_transform(None)
            np.save(filename, data)
            if 'hdf5' in dest:
                data_args = namedtuple(\
                'data_ns','hdf5 system method density slice add traj override')(\
                file,system,method,filename, ':',[],trajectory_path, True)
                add_data_driver(data_args)

                # data_args = namedtuple(\
                # 'data_ns','hdf5 system method density slice add traj override')(\
                # file,system,method,'', ':',['energy','forces'],pre['src_path'] + '/results.traj', True)
                # add_data_driver(data_args)

        if delete_workdir:
            shutil.rmtree(workdir)

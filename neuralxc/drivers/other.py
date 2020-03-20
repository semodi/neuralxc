import copy
import glob
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from pprint import pprint
from types import SimpleNamespace as SN

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from dask.distributed import Client, LocalCluster
from sklearn.base import clone
from sklearn.externals.joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import neuralxc as xc
from neuralxc.datastructures.hdf5 import *
from neuralxc.formatter import SpeciesGrouper, atomic_shape, system_shape
from neuralxc.ml import NetworkEstimator as NetworkWrapper
from neuralxc.ml import NXCPipeline
from neuralxc.ml.ensemble import ChainedEstimator, StackedEstimator
from neuralxc.ml.network import NumpyNetworkEstimator, load_pipeline
from neuralxc.ml.transformer import (GroupedPCA, GroupedStandardScaler,
                                     GroupedVarianceThreshold)
from neuralxc.ml.utils import *
from neuralxc.preprocessor import Preprocessor, driver
from neuralxc.symmetrizer import symmetrizer_factory

from .data import *

os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def plot_basis(basis):
    """ Plots a set of basis functions specified in .json file"""

    basis_instructions = json.loads(open(basis, 'r').read())
    projector = xc.projector.DensityProjector(np.eye(3), np.ones(3), basis_instructions['preprocessor'])

    for spec in basis_instructions['preprocessor']:
        if not len(spec) == 1: continue
        basis = basis_instructions['preprocessor'][spec]
        n = basis_instructions['preprocessor'][spec]['n']
        W = projector.get_W(basis)
        r = np.linspace(0, basis['r_o'], 500)
        radials = projector.radials(r, basis, W)
        for rad in radials:
            plt.plot(r, rad)
        plt.show()


def get_real_basis(atoms, basis):
    from pyscf import gto
    from ..pyscf import BasisPadder
    real_basis = {}
    for a in atoms:
        symbols = a.get_chemical_symbols()
        atom = [[s, np.array([2 * j, 0, 0])] for j, s in enumerate(symbols)]
        auxmol = gto.M(atom=atom, basis=basis)
        bp = BasisPadder(auxmol)
        padded_basis = bp.get_basis_json()
        for sym in padded_basis:
            if sym in real_basis:
                if real_basis[sym] != padded_basis[sym]:
                    raise Exception('Different basis sets across systems currently not supported')

            real_basis[sym] = padded_basis[sym]

    return real_basis


def run_engine_driver(xyz, preprocessor, workdir='.tmp/'):

    pre = json.load(open(preprocessor, 'r'))
    try:
        os.mkdir(workdir)
    except FileExistsError:
        pass

    driver(
        read(xyz, ':'),
        pre['preprocessor'].get('application', 'siesta'),
        workdir=workdir,
        nworkers=pre.get('n_workers', 1),
        kwargs=pre.get('engine_kwargs', {}))
    shutil.move(workdir + '/results.traj', './results.traj')
    if workdir == '.tmp/':
        shutil.rmtree(workdir)


def fetch_default_driver(kind, hint='', out=''):

    from collections import abc
    if hint:
        hint_cont = json.load(open(hint, 'r'))

    def nested_dict_iter(nested):
        for key, value in nested.items():
            if isinstance(value, abc.Mapping):
                yield from nested_dict_iter(value)
            else:
                yield key, value

    def find_value_in_nested(nested, truekey):
        for key, value in nested_dict_iter(nested):
            if key == truekey:
                return value
        return None

    def make_absolute(val):
        if (os.path.isfile(val) or os.path.isdir(val)) and not isinstance(val, int):
            print(os.path.isfile(val))
            print(os.path.isdir(val))
            print('DIR', val)
            val = os.path.abspath(val)
        return val

    if kind == 'pre':
        app = 'siesta'
        for key, value in nested_dict_iter(hint_cont):
            if key == 'application':
                app = value
        df_cont = json.load(open(os.path.dirname(__file__) + '/../data/pre_{}.json'.format(app), 'r'))
    else:
        df_cont = json.load(open(os.path.dirname(__file__) + '/../data/hyper.json', 'r'))

    print(df_cont)
    if hint:
        for key1 in df_cont:
            if isinstance(df_cont[key1], dict):
                for key2 in df_cont[key1]:
                    found = find_value_in_nested(hint_cont, key2)
                    if found:
                        df_cont[key1][key2] = make_absolute(found)
                    elif isinstance(df_cont[key1][key2], str):
                        df_cont[key1][key2] = make_absolute(df_cont[key1][key2])
            else:
                found = find_value_in_nested(hint_cont, key1)
                if found:
                    df_cont[key1] = make_absolute(found)
                else:
                    df_cont[key1] = make_absolute(df_cont[key1])

    if out == '':
        out = kind + '.json'

    open(out, 'w').write(json.dumps(df_cont, indent=4))


def pre_driver(xyz, srcdir, preprocessor, dest='.tmp/'):
    """ Preprocess electron densities obtained from electronic structure
    calculations
    """
    preprocessor_path = preprocessor

    pre = json.loads(open(preprocessor, 'r').read())

    atoms = read(xyz, ':')

    preprocessor = get_preprocessor(preprocessor, atoms, srcdir)
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
        print('BI', basis_instr)

        if basis_instr.get('application', 'siesta') == 'pyscf':
            real_basis = get_real_basis(atoms, basis_instr['basis'])
            for key in real_basis:
                basis_instr[key] = real_basis[key]
            pre.update({'preprocessor': basis_instr})
            open(preprocessor_path, 'w').write(json.dumps(pre))

        filename = os.path.join(workdir, basis_to_hash(basis_instr) + '.npy')
        data = preprocessor.fit_transform(None)
        np.save(filename, data)
        if 'hdf5' in dest:
            add_data_driver(hdf5=file, system=system, method=method, density=filename, add=[], traj=xyz, override=True)

            f = h5py.File(file)
            f[system].attrs.update({'species': preprocessor.species_string})
            f.close()
    if delete_workdir:
        shutil.rmtree(workdir)

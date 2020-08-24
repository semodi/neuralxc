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
import dill as pickle
from types import SimpleNamespace as SN
from .data import *
from neuralxc.preprocessor import driver
os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def plot_basis(basis):
    """ Plots a set of basis functions specified in .json file"""

    basis_instructions = json.loads(open(basis, 'r').read())
    projector = xc.projector.DensityProjector(unitcell = np.eye(3), grid=np.ones(3), basis_instructions=basis_instructions['preprocessor'])

    for spec in projector.basis:
        if not len(spec) == 1: continue
        basis = projector.basis[spec]
        if isinstance(basis, list):
            r = np.linspace(0, np.max([np.max(b_) for b in basis for b_ in b['r_o']]), 500)
        else:
            r = np.linspace(0, np.max(basis['r_o']), 500)
        W = projector.get_W(basis)
        radials = projector.radials(r, basis, W=W)
        for l, rad in enumerate(radials):
            if not isinstance(rad, list):
                rad = [rad]
            for ir, rl in enumerate(rad):
                if ir ==0:
                    plt.plot(r, rl, label='l = {}'.format(l), color='C{}'.format(l))
                else:
                    plt.plot(r, rl, color='C{}'.format(l))
        # plt.ylim(0,1)
        plt.legend()
        plt.show()


def get_real_basis(atoms, basis, spec_agnostic=False):
    from pyscf import gto
    from ..pyscf import BasisPadder
    real_basis = {}
    is_file = os.path.isfile(basis)
    # if spec_agnostic: atoms = atoms[0:1]
    if is_file:
        parsed_basis = gto.basis.parse(open(basis,'r').read())
    if spec_agnostic:
        symbols =  np.array(['O'])
    else:
        symbols = np.unique(np.array([sym for a in atoms for sym in a.get_chemical_symbols()]))

    if is_file:
        basis ={s: parsed_basis for s in symbols}
    atom = [[s, np.array([2 * j, 0, 0])] for j, s in enumerate(symbols)]

    try:
        auxmol = gto.M(atom=atom, basis=basis)
    except RuntimeError: # If spin != 0 compensate with Hydrogen
        atom += [['H', np.array([2 * len(symbols)+1, 0, 0])]]
        auxmol = gto.M(atom=atom, basis=basis)

    bp = BasisPadder(auxmol)
    padded_basis = bp.get_basis_json()
    for sym in padded_basis:
        if sym in real_basis:
            if real_basis[sym] != padded_basis[sym]:
                raise Exception('Different basis sets across systems currently not supported')

        real_basis[sym] = padded_basis[sym]

    if spec_agnostic: real_basis = {'X': real_basis['X']}
    return real_basis


def run_engine_driver(xyz, preprocessor, workdir='.tmp/'):

    pre = json.load(open(preprocessor, 'r'))
    try:
        os.mkdir(workdir)
    except FileExistsError:
        pass

    driver(read(xyz, ':'),
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
            real_basis = get_real_basis(atoms, basis_instr['basis'],
                spec_agnostic = basis_instr.get('spec_agnostic', False))
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

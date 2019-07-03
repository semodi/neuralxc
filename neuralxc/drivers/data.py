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
from .other import *
os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def add_data_driver(args):
    """ Adds data to hdf5 file"""
    try:
        file = h5py.File(args.hdf5, 'r+')
    except OSError:
        file = h5py.File(args.hdf5, 'w')

    i,j,k = [(None if a == '' else int(a)) for a in args.slice.split(':')] +\
        [None]*(3-len(args.slice.split(':')))

    ijk = slice(i, j, k)

    def obs(which):
        if which == 'energy':
            if args.traj:
                add_species(file, args.system, args.traj)
                if args.zero != None:
                    energies = np.array([a.get_potential_energy()\
                     for a in read(args.traj,':')])[ijk]
                else:
                    energies = E_from_atoms(read(args.traj,':'))
                    args.zero = 0
                add_energy(file, energies, args.system, args.method, args.override, E0=args.zero)
            else:
                raise Exception('Must provide a trajectory file')
                file.close()
        elif which == 'forces':
            if args.traj:
                add_species(file, args.system, args.traj)
                forces = [a.get_forces()\
                 for a in read(args.traj,':')]
                max_na = max([len(f) for f in forces])
                forces_padded = np.zeros([len(forces),max_na,3])
                for idx,f in enumerate(forces):
                    forces_padded[idx,:len(f)] = f
                forces = forces_padded[ijk]

                add_forces(file, forces, args.system, args.method, args.override)
            else:
                raise Exception('Must provide a trajectory file')
                file.close()
        elif which == 'density':
            add_species(file, args.system, args.traj)
            species = file[args.system].attrs['species']
            data = np.load(args.density)[ijk]
            add_density((args.density.split('/')[-1]).split('.')[0], file, data, args.system, args.method,
                        args.override)
        else:
            raise Exception('Option {} not recognized'.format(which))

    if args.density and not 'density' in args.add:
        args.add.append('density')
    for observable in args.add:
        obs(observable)

    file.close()

def merge_data_driver(args):

    if args.pre:
        pre = json.loads(open(args.pre, 'r').read())
        basis_key = basis_to_hash(pre['basis'])
    else:
        basis_key = None

    datafile = h5py.File(args.file, 'a')

    if args.optE0:
        E0 = opt_E0(datafile, args.base, args.ref)
    else:
        print('Warning: E0 is not being optimzed for merged dataset. Might produce' +\
        'unexpected behavior')

    merge_sets(datafile, args.base, basis_key, new_name = args.o +'/base', E0 = E0)
    for key in E0:
        E0[key] = 0

    merge_sets(datafile, args.ref, None, new_name = args.o +'/ref', E0 = E0)


def split_data_driver(args):
    """ Split dataset (or all data inside a group) by providing slices"""
    file = h5py.File(args.hdf5, 'r+')

    i,j,k = [(None if a == '' else int(a)) for a in args.slice.split(':')] +\
        [None]*(3-len(args.slice.split(':')))

    ijk = slice(i, j, k)

    root = args.group
    if not root[0] == '/': root = '/' + root

    def collect_all_sets(file, path):
        sets = {}
        if isinstance(file[path], h5py._hl.dataset.Dataset):
            return {path: file[path]}
        else:
            for key in file[path]:
                sets.update(collect_all_sets(file, path + '/' + key))
        return sets

    all_sets = collect_all_sets(file, root)
    split_sets = {}
    comp_sets = {}
    length = -1
    for path in all_sets:
        new_len = len(all_sets[path][:])
        if length == -1:
            length = new_len
        elif new_len != length:
            raise Exception('Datasets contained in group dont have consistent lengths')
        idx = path.find(args.group) + len(args.group)
        new_path = path[:idx] + '/' + args.label + path[idx:]
        if args.comp != '':
            idx = path.find(args.group) + len(args.group)
            comp_path = path[:idx] + '/' + args.comp + path[idx:]
            comp_sets[comp_path] = all_sets[path][:].tolist()
            del comp_sets[comp_path][ijk]
        split_sets[new_path] = all_sets[path][ijk]

    for new_path in split_sets:
        file.create_dataset(new_path, data=split_sets[new_path])

    for new_path, path in zip(split_sets, all_sets):
        file['/'.join(new_path.split('/')[:-1])].attrs.update(file['/'.join(path.split('/')[:-1])].attrs)
    if comp_sets:
        for comp_path in comp_sets:
            file.create_dataset(comp_path, data=comp_sets[comp_path])
        for new_path, path in zip(comp_sets, all_sets):
            file['/'.join(new_path.split('/')[:-1])].attrs.update(file['/'.join(path.split('/')[:-1])].attrs)
    # print(split_sets)


def delete_data_driver(args):
    """ Deletes data in hdf5 file"""
    file = h5py.File(args.hdf5, 'r+')
    root = args.group
    if not root[0] == '/': root = '/' + root
    del file[root]


def sample_driver(args):
    """ Given a dataset, perform sampling in feature space"""

    preprocessor = args.preprocessor
    hdf5 = args.hdf5

    pre = json.loads(open(preprocessor, 'r').read())

    datafile = h5py.File(hdf5[0], 'r')
    basis_key = basis_to_hash(pre['basis'])
    data = load_sets(datafile, hdf5[1], hdf5[1], basis_key, args.cutoff)
    basis = pre['basis']
    symmetrizer_instructions = {'symmetrizer_type': 'casimir'}
    symmetrizer_instructions.update({'basis': basis})
    species = [''.join(find_attr_in_tree(datafile, hdf5[1], 'species'))]
    spec_group = SpeciesGrouper(basis, species)
    symmetrizer = symmetrizer_factory(symmetrizer_instructions)

    sampler_pipeline = get_default_pipeline(basis, species, pca_threshold=1)
    sampler_pipeline = Pipeline(sampler_pipeline.steps)
    sampler_pipeline.steps[-1] = ('sampler', SampleSelector(args.size))
    sampler_pipeline.fit(data)
    sample = sampler_pipeline.predict(data)
    np.save(args.dest, np.array(sample).flatten())

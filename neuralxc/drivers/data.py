import json
import os

import h5py
import numpy as np
from ase.io import read
from sklearn.pipeline import Pipeline

from neuralxc.datastructures.hdf5 import *
from neuralxc.formatter import make_nested_absolute
from neuralxc.ml.utils import *
from neuralxc.utils import ConfigFile
# from .other import *
__all__ = ['add_data_driver', 'merge_data_driver', 'split_data_driver', 'delete_data_driver', 'sample_driver']

bi_slice = slice
os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def add_data_driver(hdf5, system, method, add, traj='', density='', override=False, slice=':', zero=None, addto=''):
    """ Adds data to hdf5 file"""
    try:
        file = h5py.File(hdf5, 'r+')
        file.close()
    except OSError:
        file = h5py.File(hdf5, 'w')
        file.close()

    with h5py.File(hdf5, 'r+') as file:
        i,j,k = [(None if a == '' else int(a)) for a in slice.split(':')] +\
            [None]*(3-len(slice.split(':')))

        ijk = bi_slice(i, j, k)

        def obs(which, zero):
            if which == 'energy':
                if traj:
                    try:
                        read(traj)
                        add_species(file, system, traj)
                        if zero is not None:
                            energies = np.array([a.get_potential_energy()\
                             for a in read(traj,':')])[ijk]
                        else:
                            energies = E_from_atoms(read(traj, ':'))
                            zero = 0
                    except ValueError:
                        energies = np.load(traj)

                    if addto:
                        energies0 = file[addto][:]
                        energies += energies0
                    add_energy(file, energies, system, method, override, E0=zero)
                else:
                    raise Exception('Must provide a either trajectory file or .npy file containing energies')
            elif which == 'forces':
                if traj:
                    add_species(file, system, traj)
                    forces = [a.get_forces()\
                     for a in read(traj,':')]
                    max_na = max([len(f) for f in forces])
                    forces_padded = np.zeros([len(forces), max_na, 3])
                    for idx, f in enumerate(forces):
                        forces_padded[idx, :len(f)] = f
                    forces = forces_padded[ijk]

                    add_forces(file, forces, system, method, override)
                else:
                    raise Exception('Must provide a trajectory file')
            elif which == 'density':
                add_species(file, system, traj)
                data = np.load(density)[ijk]
                add_density((density.split('/')[-1]).split('.')[0], file, data, system, method, override)
            else:
                raise Exception('Option {} not recognized'.format(which))

        if density and not 'density' in add:
            add.append('density')
        for observable in add:
            obs(observable, zero)


def merge_data_driver(file, base, ref, out, optE0=False, pre=''):

    if pre:
        pre = ConfigFile(pre)
        basis_key = basis_to_hash(pre['basis'])
    else:
        basis_key = None

    datafile = h5py.File(file, 'a')

    if optE0:
        E0 = opt_E0(datafile, base, ref)
    else:
        print('Warning: E0 is not being optimzed for merged dataset. Might produce' +\
        'unexpected behavior')

    merge_sets(datafile, base, basis_key, new_name=out + '/base', E0=E0)
    for key in E0:
        E0[key] = 0

    merge_sets(datafile, ref, None, new_name=out + '/ref', E0=E0)


def split_data_driver(hdf5, group, label, slice=':', comp=''):
    """ Split dataset (or all data inside a group) by providing slices"""
    file = h5py.File(hdf5, 'r+')

    i,j,k = [(None if a == '' else int(a)) for a in slice.split(':')] +\
        [None]*(3-len(slice.split(':')))

    ijk = bi_slice(i, j, k)

    root = group
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
        idx = path.find(group) + len(group)
        new_path = path[:idx] + '/' + label + path[idx:]
        if comp != '':
            idx = path.find(group) + len(group)
            comp_path = path[:idx] + '/' + comp + path[idx:]
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


def delete_data_driver(hdf5, group):
    """ Deletes data in hdf5 file"""
    file = h5py.File(hdf5, 'r+')
    root = group
    if not root[0] == '/': root = '/' + root
    del file[root]


def sample_driver(preprocessor, size, hdf5, dest='sample.npy', cutoff=0.0):
    """ Given a dataset, perform sampling in feature space"""

    pre = make_nested_absolute(ConfigFile(preprocessor))

    datafile = h5py.File(hdf5[0], 'r')
    basis = pre['preprocessor']
    basis_key = basis_to_hash(basis)
    data = load_sets(datafile, hdf5[1], hdf5[1], basis_key, cutoff)
    symmetrizer_instructions = {'symmetrizer_type': pre.get('symmetrizer_type', 'trace')}
    symmetrizer_instructions.update({'basis': basis})
    species = [''.join(find_attr_in_tree(datafile, hdf5[1], 'species'))]

    sampler_pipeline = get_default_pipeline(basis,
                                            species,
                                            symmetrizer_type=symmetrizer_instructions['symmetrizer_type'],
                                            pca_threshold=1)

    sampler_pipeline = Pipeline(sampler_pipeline.steps)
    sampler_pipeline.steps[-1] = ('sampler', SampleSelector(size))
    sampler_pipeline.fit(data)
    sample = sampler_pipeline.predict(data)
    np.save(dest, np.array(sample).flatten())

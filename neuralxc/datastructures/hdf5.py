from ase.io import read, write
import hashlib
import json
import numpy as np
import neuralxc.ml.utils


def add_energy(*args, **kwargs):
    return add_data('energy', *args, **kwargs)


def add_forces(*args, **kwargs):
    return add_data('forces', *args, **kwargs)


def add_density(key, *args, **kwargs):
    return add_data(key, *args, **kwargs)


def add_species(file, system, traj_path=''):
    """
    Add an attribute containing the species string for a given
    system (ex. for water: {'species': 'OHH'})

    Parameters
    ----------

    file: hdf5 file handle
        File to add data to
    system: str
        System label defining first part of group
        in datafile
    """

    order = [system]
    cg = file  #Current group
    for idx, o in enumerate(order):
        if not o in cg.keys():
            cg = cg.create_group(o)
        else:
            cg = cg[o]

    if not 'species' in cg.attrs:
        if not traj_path:
            raise Exception('Must provide a trajectory file to define species')

        species = {}
        for atoms in read(traj_path, ':'):
            species[''.join(atoms.get_chemical_symbols())] = 0
        species = ''.join([key for key in species])

        cg.attrs.update({'species': species})


def add_data(which, file, data, system, method, override=False, E0=None):
    """
    Add data to hdf5 file.

    Parameters
    ----------

    which: str
        Add 'energy', 'forces' or 'density'
    file: hdf5 file handle
        File to add data to
    data: numpy ndarray
        Data to add
    system: str
        System label defining first part of group
        in datafile
    method: str
        Method label defining second part of group
        in datafile
    override: bool
        If dataset already exists in file, override it?
    """

    order = [system, method]
    if not which in ['energy', 'forces']:
        order.append('density')

    cg = file  #Current group
    for idx, o in enumerate(order):
        if not o in cg.keys():
            cg = cg.create_group(o)
        else:
            cg = cg[o]

    if which == 'energy':
        if E0 == None:
            cg.attrs.update({'E0': min(data)})
        else:
            cg.attrs.update({'E0': E0})

    print('{} systems found, adding {}'.format(len(data), which))

    def create_dataset():
        cg.create_dataset(which, data=data)

    try:
        create_dataset()
    except RuntimeError:
        if override:
            del cg[which]
            create_dataset()
        else:
            print('Already exists. Set override=True')


def merge_sets(file, datasets, density_key=None, new_name='merged', E0={}):

    energies = [file[data + '/energy'][:] for data in datasets]
    if not E0:
        energies = [e - nxc.ml.utils.find_attr_in_tree(file, data, 'E0') for e, data in zip(energies, datasets)]

    forces_found = True
    try:
        forces = [file[data + '/forces'][:] for data in datasets]
    except KeyError:
        forces_found = False

    if density_key:
        densities = [file[data + '/density/' + density_key][:] for data in datasets]

        densities_full = np.zeros(
            [sum([len(d) for d in densities]), sum([d.shape[1] for d in densities])])
        line_mark = 0
        col_mark = 0
        for d in densities:
            densities_full[line_mark:line_mark + d.shape[0], col_mark:col_mark + d.shape[1]] = d
            line_mark += d.shape[0]
            col_mark += d.shape[1]

    if forces_found:
        forces_full = np.zeros([sum([len(d) for d in forces]), max([d.shape[1] for d in forces]), 3])
        line_mark = 0

        for f in forces:
            forces_full[line_mark:line_mark + f.shape[0], :f.shape[1]] = f
            line_mark += f.shape[0]

    species = [neuralxc.ml.utils.find_attr_in_tree(file, data, 'species') for data in datasets]
    if E0:
        energies = [
            e - sum([s.count(element) * value for element, value in E0.items()]) for e, s in zip(energies, species)
        ]
    species = [''.join(species)]
    energies = np.concatenate(energies)

    if forces_found:
        assert len(energies) == len(forces_full)
    if density_key:
        assert len(energies) == len(densities_full)
    try:
        file.create_group(new_name)
    except ValueError:
        del file[new_name]
        file.create_group(new_name)
    file[new_name].attrs.update({'species': species})
    file[new_name].attrs.update({'E0': 0})
    file.create_dataset(new_name + '/energy', data=energies)
    if forces_found:
        file.create_dataset(new_name + '/forces', data=forces_full)
    if density_key:
        file.create_dataset(new_name + '/density/' + density_key, data=densities_full)


def basis_to_hash(basis):
    """
    Convert a given basis to a unique identifier

    Parameters
    ---------

    basis: dict
        Contains the basis like so : {'species1': {'n': 1, 'l': 2}...}

    Returns
    --------

    hash: str
        Encoding of the basis set
    """
    return hashlib.md5(json.dumps(basis).encode()).hexdigest()

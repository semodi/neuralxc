import dask
import dask.distributed
from dask.distributed import Client, LocalCluster
import sys
import argparse
import os
from ase.io import read, write
import json
import itertools
from ase import Atoms
import numpy as np
from ..engines import Engine


def in_private_dir(method):
    def wrapper_private_dir(dir, *args, **kwargs):
        try:
            os.chdir(dir)
        except FileNotFoundError:
            os.mkdir(dir)
            os.chdir(dir)
        return method(*args, **kwargs)

    return wrapper_private_dir


@in_private_dir
def calculate_system(atoms, app, kwargs):
    eng = Engine(app, **kwargs)
    atoms = eng.compute(atoms)
    return atoms


def mbe_driver(atoms, app, workdir, kwargs, nworkers):
    """ Many-body expansion"""

    building_block = kwargs.get('mbe_block', 'OHH')
    n_block = len(building_block)

    results = calculate_distributed(atoms, app, workdir, kwargs, nworkers)
    species = [a.get_chemical_symbols() for a in atoms]
    n_mol = int(len(species[0]) / n_block)
    for s in species:
        n_mol_new = int(len(s) / n_block)
        if not n_mol == n_mol_new:
            raise Exception('Every snapshot in trajectory must contain same number of molecules')
        if not s == [s for s in building_block] * int(len(s) / n_block):
            print(s)
            raise Exception('Trajectory file must contain atoms in the oder OHHOHH...')


# if workdir:
#     mbe_root = dir[:-len(workdir)]
# else:
    mbe_root = workdir

    lower_results = []

    for n in range(1, n_mol):
        new_atoms = [
            Atoms(
                building_block * n,
                positions=a.get_positions().reshape(-1, n_block, 3)[np.array(comb)].reshape(-1, 3),
                pbc=a.get_pbc(),
                cell=a.get_cell()) for a in atoms for comb in itertools.combinations(range(n_mol), n)
        ]
        try:
            os.mkdir(mbe_root + '/mbe_{}'.format(n))
        except FileExistsError:
            pass
        lower_results.append(calculate_distributed(new_atoms, app, mbe_root + '/mbe_{}'.format(n), kwargs, nworkers))

    etot = np.array([a.get_potential_energy() for a in results])
    for i, lr in enumerate(lower_results[::-1]):
        write(mbe_root + '/mbe_{}/results.traj'.format(n_mol - (i + 1)), lr)
        epart = np.array([((-1)**(i + 1)) * a.get_potential_energy() for a in lr]).reshape(len(etot), -1)
        epart = np.sum(epart, axis=-1)
        etot += epart

    for i, e in enumerate(etot):
        results[i].calc.results['energy'] = e
        results[i].calc.results['forces'] = None

    return results


def calculate_distributed(atoms, app, workdir, kwargs, n_workers=-1):

    cwd = os.getcwd()
    if n_workers > 1:
        print('Calculating {} systems on'.format(len(atoms)))
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        print(cluster)
        client = Client(cluster)
        my_map = client.map
        get_result = lambda x: x.result()
    else:
        my_map = map
        get_result = lambda x: x

    futures = my_map(calculate_system, [os.path.join(workdir, str(i)) for i, _ in enumerate(atoms)], atoms,
                     [app] * len(atoms), [kwargs] * len(atoms))

    results = [get_result(f) for f in futures]
    os.chdir(cwd)
    return results


def driver(atoms, app, workdir, nworkers, kwargs):
    dir = os.path.abspath(workdir)
    # results = calculate_distributed(atoms, app, dir, kwargs, nworkers)
    if kwargs.get('mbe', False):
        results = mbe_driver(atoms, app, dir, kwargs, nworkers)
    else:
        results = calculate_distributed(atoms, app, dir, kwargs, nworkers)
    results_path = os.path.join(dir, 'results.traj')
    write(results_path, results)
    return results

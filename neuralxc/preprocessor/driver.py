"""
driver.py
Contains driver functions that 'applies' an Engine (see ../engines/), i.e.
an electronic structure code that computes energy-density pairs, to a dataset
of structures. If desired this can be done in a distributed fashion using Dask.
"""

import itertools
import os

import numpy as np
from ase import Atoms
from ase.io import write
from dask.distributed import Client, LocalCluster

from neuralxc.engines import Engine

# def in_private_dir(method):
#     def wrapper_private_dir(dir, *args, **kwargs):
#         try:
#             os.chdir(dir)
#         except FileNotFoundError:
#             os.mkdir(dir)
#             os.chdir(dir)
#         return method(*args, **kwargs)
#
#     return wrapper_private_dir


def calculate_system(dir, atoms, app, kwargs):
    try:
        os.chdir(dir)
    except FileNotFoundError:
        os.mkdir(dir)
        os.chdir(dir)
    eng = Engine(app, **kwargs)
    atoms = eng.compute(atoms)
    return atoms


def mbe_driver(atoms, app, workdir, kwargs, nworkers):
    """ Many-body expansion"""

    building_block = kwargs.get('mbe_block', 'OHH')
    n_block = len(building_block)

    results = calculate_distributed(atoms, app, workdir, kwargs, nworkers)
    species = [a.get_chemical_symbols() for a in atoms]
    n_mol = len(species[0]) // n_block
    for s in species:
        n_mol_new = len(s) // n_block
        if n_mol != n_mol_new:
            raise Exception('Every snapshot in trajectory must contain same number of molecules')
        if s != list(building_block) * (len(s) // n_block):
            print(s)
            raise Exception('Trajectory file must contain atoms in the oder OHHOHH...')

    mbe_root = workdir

    lower_results = []

    for n in range(1, n_mol):
        new_atoms = [
            Atoms(building_block * n,
                  positions=a.get_positions().reshape(-1, n_block, 3)[np.array(comb)].reshape(-1, 3),
                  pbc=a.get_pbc(),
                  cell=a.get_cell()) for a in atoms for comb in itertools.combinations(range(n_mol), n)
        ]
        try:
            os.mkdir(mbe_root + f'/mbe_{n}')
        except FileExistsError:
            pass
        lower_results.append(
            calculate_distributed(
                new_atoms, app, mbe_root + f'/mbe_{n}', kwargs, nworkers
            )
        )


    etot = np.array([a.get_potential_energy() for a in results])
    for i, lr in enumerate(lower_results[::-1]):
        write(mbe_root + f'/mbe_{n_mol - (i + 1)}/results.traj', lr)
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
        print(f'Calculating {len(atoms)} systems on')
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
    """
    Applies app (Engine) across dataset of structures.

    Parameters
    -----------
    atoms, list of ase.Atoms
        Dataset containing structures
    app, Engine
        Engine (see ../engines/) controlling the elecronic structure code
    wordir, str
        Name of work directory
    nworkers, int
        Number of workers for Dask cluster

    Returns
    --------
    results, list of ase.Atoms
        Original dataset passed as "atoms" but with total energy saved
        as single point calculator attribute (can be
        accessed as atoms.get_potential_energy())
    """

    dir = os.path.abspath(workdir)
    # results = calculate_distributed(atoms, app, dir, kwargs, nworkers)
    if kwargs.get('mbe', False):
        results = mbe_driver(atoms, app, dir, kwargs, nworkers)
    else:
        results = calculate_distributed(atoms, app, dir, kwargs, nworkers)
    results_path = os.path.join(dir, 'results.traj')
    write(results_path, results)
    return results

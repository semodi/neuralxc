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
    results = calculate_distributed(atoms, app, dir, kwargs, nworkers)
    results_path = os.path.join(dir, 'results.traj')
    write(results_path, results)
    return results

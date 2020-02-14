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
from .data import *
from .other import *
from neuralxc.preprocessor import driver
from glob import glob
os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def mkdir(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass


def shcopy(src, dest):
    try:
        shutil.copy(src, dest)
    except FileExistsError:
        pass


def shcopytree(src, dest):
    try:
        shutil.copytree(src, dest)
    except FileExistsError:
        pass


def create_report(path='.'):
    paths = glob('*/stat*')

    stats = []
    for p in paths:
        stats.append(json.load(open(p, 'r')))
        stats[-1]['path'] = p

    stats = pd.DataFrame(stats)

    stats['kind'] = stats['path'].apply(lambda x: x.split('/')[1].split('_')[1])
    stats['it'] = stats['path'].apply(lambda x: x.split('/')[0].strip('it'))

    stats.loc[stats['kind'] == 'test', 'it'] = 999
    stats['it'] = stats['it'].astype(int)
    stats.loc[stats['kind'] == 'fit', 'it'] += 1
    stats = stats.sort_values(['it', 'kind'])
    stats.loc[stats['kind'] == 'test', 'it'] = -1
    stats = stats.reset_index(drop=True)
    stats = stats.drop(len(stats) - 2, axis=0).drop('path', axis=1).drop('mean deviation', axis=1)

    stats.columns = [{'it': 'Iteration', 'kind': 'Kind'}.get(x, x.upper() + ' (eV)') for x in stats.columns]
    stats.loc[stats['Kind'] == 'test', 'Iteration'] = ''
    stats.index = [k + '_' + str(it) for k, it in zip(stats['Kind'], stats['Iteration'])]
    stats.index = [{'test_': 'test'}.get(i, i) for i in stats.index]
    stats['Kind'] = stats['Kind'].apply(lambda x: {'sc': 'Self-consistent', 'fit': 'Fitting', 'test': 'Testing'}[x])

    return stats


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


def convert_tf(tf_path, np_path):
    """ Converts the tensorflow estimator inside a NXCPipeline to a simple
    numpy base estimator"""

    nxc_tf = xc.NeuralXC(tf_path)
    pipeline = nxc_tf._pipeline

    C = {}
    basis = pipeline.get_basis_instructions()
    if not basis.get('spec_agnostic', False):
        basis.pop('X',None)

    for sym in basis:
        if len(sym) > 2: continue
        C[sym] = np.zeros([1, 1, basis[sym]['n'] * basis[sym]['l']**2])
    D = nxc_tf.symmetrizer.get_symmetrized(C)
    nxc_tf._pipeline.predict(D)
    nxc_tf._pipeline.save(np_path, True, True)


def merge_driver(chained, merged):
    """ Converts the tensorflow estimator inside a NXCPipeline to a simple
    numpy base estimator"""

    nxc_tf = xc.NeuralXC(chained)
    pipeline = nxc_tf._pipeline

    label, estimator = pipeline.steps[-1]
    _, npestimator = pipeline.steps[-2]

    if not isinstance(npestimator, NumpyNetworkEstimator):
        raise Exception('Something went wrong. Second to last pipeline element'\
        +' must be NumpyNetworkEstimator')

    if not isinstance(estimator, NumpyNetworkEstimator):
        if not isinstance(estimator, NetworkWrapper):
            raise Exception('Something went wrong. Last pipeline element'\
            +' must be an estimator')
        else:
            convert_tf(tf_path=chained, np_path=merged)
            chained = merged

            nxc_tf = xc.NeuralXC(chained)
            pipeline = nxc_tf._pipeline

            label, estimator = pipeline.steps[-1]
            _, npestimator = pipeline.steps[-2]

    if not npestimator.trunc:
        npestimator = npestimator.trunc_after(-1)

    pipeline.steps[-2] = (label, ChainedEstimator([npestimator, estimator]).merge())
    pipeline.steps = pipeline.steps[:-1]
    nxc_tf._pipeline.save(merged, True, True)


def adiabatic_driver(xyz,
                     preprocessor,
                     hyper,
                     data='',
                     hyper2='',
                     maxit=5,
                     tol=0.0005,
                     b0=1,
                     b_decay=0.1,
                     hotstart=0,
                     sets='',
                     nozero=False,
                     model0='',
                     fullstack=False,
                     hyperopt=False,
                     max_epochs=0,
                     scale_targets=False,
                     scale_exp=2):

    statistics_sc = {'mae': 1000}
    xyz = os.path.abspath(xyz)
    pre = json.loads(open(preprocessor, 'r').read())
    engine_kwargs = pre.get('engine_kwargs', {})
    if sets:
        sets = os.path.abspath(sets)

    if model0:
        convert_tf(tf_path=model0, np_path='model0')
        model0 = 'model0'
        model0 = os.path.abspath(model0)
        engine_kwargs = {'nxc': model0}
        engine_kwargs.update(pre.get('engine_kwargs', {}))
        ensemble = True
        if fullstack:
            model0_orig = model0
            model0 = ''
        else:
            model0_orig = model0
    else:
        ensemble = False
        model0 = ''

    if nozero:
        E0 = 0
    else:
        E0 = None
    b = b0
    iteration = 0


    if hotstart == 0:
        if data:
            mkdir('it0')
            shcopy(data, 'it0/data.hdf5')
            shcopy(preprocessor, 'it0/pre.json')
            shcopy(hyper, 'it0/hyper.json')
            os.chdir('it0')
            open('sets.inp', 'w').write('data.hdf5 \n system/base \t system/ref')
            if sets:
                open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
            statistics_sc = \
            eval_driver(hdf5=['data.hdf5','system/base','system/ref'])

            open('statistics_sc', 'w').write(json.dumps(statistics_sc))
            statistics_fit = fit_driver(
                preprocessor='pre.json',
                hyper='hyper.json',
                model=model0,
                ensemble=ensemble,
                sets='sets.inp',
                hyperopt=True,
                b=b,
                target_scale = \
                    (1 - ((maxit - 3  - iteration)/(maxit - 2))**scale_exp if scale_targets else 1))
            open('statistics_fit', 'w').write(json.dumps(statistics_fit))
            convert_tf(tf_path='best_model', np_path='merged_new')
            os.chdir('../')
        else:
            print('====== Iteration {} ======'.format(iteration))
            if ensemble and not fullstack:
                shcopytree(model0, 'it0/nxc')
            mkdir('it{}'.format(iteration))
            shcopy(preprocessor, 'it{}/pre.json'.format(iteration))
            shcopy(hyper, 'it{}/hyper.json'.format(iteration))
            os.chdir('it{}'.format(iteration))
            open('sets.inp', 'w').write('data.hdf5 \n system/it{} \t system/ref'.format(iteration))
            if sets:
                open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
            mkdir('workdir')
            driver(
                read(xyz, ':'),
                pre['preprocessor'].get('application', 'siesta'),
                workdir='workdir',
                nworkers=pre.get('n_workers', 1),
                kwargs=engine_kwargs)
            pre_driver(xyz, 'workdir', preprocessor='pre.json', dest='data.hdf5/system/it{}'.format(iteration))
            add_data_driver(
                hdf5='data.hdf5',
                system='system',
                method='it0',
                add=['energy'],
                traj='workdir/results.traj',
                override=True,
                zero=E0)
            add_data_driver(
                hdf5='data.hdf5', system='system', method='ref', add=['energy'], traj=xyz, override=True, zero=E0)
            statistics_sc = \
            eval_driver(hdf5=['data.hdf5','system/it{}'.format(iteration),
                    'system/ref'])

            open('statistics_sc', 'w').write(json.dumps(statistics_sc))
            statistics_fit = fit_driver(
                preprocessor='pre.json',
                hyper='hyper.json',
                model=model0,
                ensemble=ensemble,
                sets='sets.inp',
                b=b,
                hyperopt=hyperopt,
                target_scale = \
                        (1 - ((maxit - 3  - iteration)/(maxit - 2))**scale_exp if scale_targets else 1))
            if hyperopt:
                if scale_targets:
                    raise Exception('Hyperpar optimization and scale_targets currently not supported')
                shcopy('hyper.json', 'hyper_old.json')
                best_pars = json.load(open('best_params.json', 'r'))
                if max_epochs > 0:
                    best_pars['hyperparameters']['estimator__max_steps'] = max_epochs
                with open('hyper.json', 'w') as file:
                    file.write(json.dumps(best_pars, indent=4))

                statistics_fit = fit_driver(
                    preprocessor='pre.json', hyper='hyper.json', model=model0, ensemble=ensemble, sets='sets.inp', b=b)

            open('statistics_fit', 'w').write(json.dumps(statistics_fit))

            os.chdir('../')
        hotstart += 1
    it_label = 0
    if not hotstart == -1:
        for it in range(hotstart, maxit + 1):
            b = b * b_decay
            iteration = 0
            print('====== Iteration {} ======'.format(it_label))
            # mkdir('it{}'.format(iteration))
            # shcopy(preprocessor, 'it{}/pre.json'.format(iteration))
            # shcopy(hyper, 'it{}/hyper.json'.format(iteration))
            os.chdir('it{}'.format(iteration))
            open('sets.inp', 'w').write('data.hdf5 \n *system/it{} \t system/ref'.format(iteration))

            if ensemble:
                convert_tf('best_model', 'best_model_np')
                ensemble_driver(dest='merged', models=[model0_orig, 'best_model_np'], estonly=not fullstack)
            else:
                shcopytree('best_model', 'merged')

            if sets:
                open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
            shutil.rmtree('workdir')
            mkdir('workdir')
            engine_kwargs = {'nxc': '../../merged'}
            engine_kwargs.update(pre.get('engine_kwargs', {}))
            shcopytree('best_model', 'model_it{}'.format(it_label))

            driver(
                read(xyz, ':'),
                pre['preprocessor'].get('application', 'siesta'),
                workdir='workdir',
                nworkers=pre.get('n_workers', 1),
                kwargs=engine_kwargs)
            pre_driver(xyz, 'workdir', preprocessor='pre.json', dest='data.hdf5/system/it{}'.format(iteration))

            add_data_driver(
                hdf5='data.hdf5',
                system='system',
                method='it{}'.format(iteration),
                add=['energy'],
                traj='workdir/results.traj',
                override=True,
                zero=E0)
            old_statistics = dict(statistics_sc)
            statistics_sc = \
            eval_driver(hdf5=['data.hdf5','system/it{}'.format(iteration),
                    'system/ref'])

            open('statistics_sc', 'a').write('\n' + json.dumps(statistics_sc))
            open('model_it{}/statistics_sc'.format(it_label), 'w').write('\n' + json.dumps(statistics_sc))

            print(json.load(open('hyper.json', 'r')))
            statistics_fit = fit_driver(
                preprocessor='pre.json', hyper='hyper.json', model='best_model', sets='sets.inp', b=b,
                target_scale = min(1,(1 - ((maxit - 3  - it)/(maxit - 2))**scale_exp if scale_targets else 1)))

            open('statistics_fit', 'a').write('\n' + json.dumps(statistics_fit))
            os.chdir('../')
            it_label += 1
        else:
            print('Maximum number of iterations reached. Proceeding to test set...')

    print('====== Testing ======'.format(iteration))
    mkdir('testing')

    shcopy('it{}/data.hdf5'.format(iteration), 'testing/data.hdf5')
    shcopytree('it{}/best_model'.format(iteration), 'testing/nxc')
    os.chdir('testing')
    mkdir('workdir')

    engine_kwargs = {'nxc': '../../nxc'}
    engine_kwargs.update(pre.get('engine_kwargs', {}))
    driver(
        read('../testing.traj', ':'),
        pre['preprocessor'].get('application', 'siesta'),
        workdir='workdir',
        nworkers=pre.get('n_workers', 1),
        kwargs=engine_kwargs)
    add_data_driver(
        hdf5='data.hdf5',
        system='system',
        method='testing/ref',
        add=['energy'],
        traj='../testing.traj',
        override=True,
        zero=E0)
    add_data_driver(
        hdf5='data.hdf5',
        system='system',
        method='testing/nxc',
        add=['energy'],
        traj='workdir/results.traj',
        override=True,
        zero=E0)

    statistics_test = eval_driver(hdf5=['data.hdf5', 'system/testing/nxc', 'system/testing/ref'])
    open('statistics_test', 'w').write(json.dumps(statistics_test))


def workflow_driver(xyz,
                    preprocessor,
                    hyper,
                    data='',
                    hyper2='',
                    maxit=5,
                    tol=0.0005,
                    hotstart=0,
                    sets='',
                    nozero=False,
                    model0='',
                    fullstack=False,
                    stop_early=True):
    statistics_sc = {'mae': 1000}
    xyz = os.path.abspath(xyz)
    pre = json.loads(open(preprocessor, 'r').read())
    engine_kwargs = pre.get('engine_kwargs', {})
    if sets:
        sets = os.path.abspath(sets)

    if model0:
        model0 = os.path.abspath(model0)
        engine_kwargs = {'nxc': model0}
        engine_kwargs.update(pre.get('engine_kwargs', {}))
        ensemble = True
        if fullstack:
            model0 = ''
        else:
            model0 = model0
    else:
        ensemble = False
        model0 = ''

    if nozero:
        E0 = 0
    else:
        E0 = None
    iteration = 0
    if hotstart == 0:
        if data:
            mkdir('it0')
            shcopy(data, 'it0/data.hdf5')
            shcopy(preprocessor, 'it0/pre.json')
            shcopy(hyper, 'it0/hyper.json')
            os.chdir('it0')
            open('sets.inp', 'w').write('data.hdf5 \n system/base \t system/ref')
            if sets:
                open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
            statistics_sc = \
            eval_driver(hdf5=['data.hdf5','system/base',
                    'system/ref'])

            open('statistics_sc', 'w').write(json.dumps(statistics_sc))
            statistics_fit = fit_driver(
                preprocessor='pre.json',
                hyper='hyper.json',
                model=model0,
                ensemble=ensemble,
                sets='sets.inp',
                hyperopt=True)
            open('statistics_fit', 'w').write(json.dumps(statistics_fit))
            convert_tf(tf_path='best_model', np_path='merged_new')
            os.chdir('../')
        else:
            print('====== Iteration {} ======'.format(iteration))
            if ensemble and not fullstack:
                shcopytree(model0, 'it0/nxc')
            mkdir('it{}'.format(iteration))
            shcopy(preprocessor, 'it{}/pre.json'.format(iteration))
            shcopy(hyper, 'it{}/hyper.json'.format(iteration))
            os.chdir('it{}'.format(iteration))
            open('sets.inp', 'w').write('data.hdf5 \n system/it{} \t system/ref'.format(iteration))
            if sets:
                open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
            mkdir('workdir')
            driver(
                read(xyz, ':'),
                pre['preprocessor'].get('application', 'siesta'),
                workdir='workdir',
                nworkers=pre.get('n_workers', 1),
                kwargs=engine_kwargs)
            pre_driver(xyz, 'workdir', preprocessor='pre.json', dest='data.hdf5/system/it{}'.format(iteration))
            add_data_driver(
                hdf5='data.hdf5',
                system='system',
                method='it0',
                add=['energy'],
                traj='workdir/results.traj',
                override=True,
                zero=E0)
            add_data_driver(
                hdf5='data.hdf5', system='system', method='ref', add=['energy'], traj=xyz, override=True, zero=E0)
            statistics_sc = \
            eval_driver(hdf5=['data.hdf5','system/it{}'.format(iteration),
                    'system/ref'])

            open('statistics_sc', 'w').write(json.dumps(statistics_sc))
            statistics_fit = fit_driver(
                preprocessor='pre.json',
                hyper='hyper.json',
                model=model0,
                ensemble=ensemble,
                sets='sets.inp',
                hyperopt=True)

            open('statistics_fit', 'w').write(json.dumps(statistics_fit))
            convert_tf(tf_path='best_model', np_path='merged_new')

            os.chdir('../')
        hotstart += 1
    if not hotstart == -1:
        for iteration in range(hotstart, maxit + 1):
            print('====== Iteration {} ======'.format(iteration))
            mkdir('it{}'.format(iteration))
            shcopy('it{}/data.hdf5'.format(iteration - 1), 'it{}/data.hdf5'.format(iteration))
            shcopy(preprocessor, 'it{}/pre.json'.format(iteration))
            if hyper2:
                shcopy(hyper2, 'it{}/hyper.json'.format(iteration))
            else:
                hyp = json.load(open(hyper, 'r'))
                hyp['hyperparameters']['estimator__b'] = 0
                json.dump(hyp, open('it{}/hyper.json'.format(iteration), 'w'), indent=4)

            shcopytree('it{}/merged_new'.format(iteration - 1), 'it{}/merged'.format(iteration))
            os.chdir('it{}'.format(iteration))
            if ensemble:
                ensemble_driver(dest='nxc', models=[model0, 'merged'], estonly=not fullstack)
            else:
                shcopytree('merged', 'nxc')

            open('sets.inp', 'w').write('data.hdf5 \n *system/it{} \t system/ref'.format(iteration))
            if sets:
                open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
            mkdir('workdir')
            engine_kwargs = {'nxc': '../../nxc'}
            engine_kwargs.update(pre.get('engine_kwargs', {}))
            driver(
                read(xyz, ':'),
                pre['preprocessor'].get('application', 'siesta'),
                workdir='workdir',
                nworkers=pre.get('n_workers', 1),
                kwargs=engine_kwargs)
            pre_driver(xyz, 'workdir', preprocessor='pre.json', dest='data.hdf5/system/it{}'.format(iteration))

            add_data_driver(
                hdf5='data.hdf5',
                system='system',
                method='it{}'.format(iteration),
                add=['energy'],
                traj='workdir/results.traj',
                override=True,
                zero=E0)
            old_statistics = dict(statistics_sc)
            statistics_sc = \
            eval_driver(hdf5=['data.hdf5','system/it{}'.format(iteration),
                    'system/ref'])

            open('statistics_sc', 'w').write(json.dumps(statistics_sc))

            if old_statistics['mae'] - statistics_sc['mae'] < tol and stop_early:
                if old_statistics['mae'] - statistics_sc['mae'] < 0:
                    print('Self-consistent error increased in this iteration: dMAE = {} eV'.format(
                        old_statistics['mae'] - statistics_sc['mae']))
                    iteration -= 1
                    os.chdir('../')
                else:
                    print('Iterative training converged: dMAE = {} eV'.format(
                        old_statistics['mae'] - statistics_sc['mae']))
                    os.chdir('../')
                break

            chain_driver(hyper='hyper.json', model='merged', dest='chained')
            statistics_fit = fit_driver(
                preprocessor='pre.json', hyper='hyper.json', model='chained', sets='sets.inp', hyperopt=True)

            open('statistics_fit', 'w').write(json.dumps(statistics_fit))
            if statistics_fit['mae'] > statistics_sc['mae'] and stop_early:
                print('Stopping iterative training because fitting error is larger than self-consistent error')
                os.chdir('../')
                break
            merge_driver(chained='best_model', merged='merged_new')

            os.chdir('../')
        else:
            print('Maximum number of iterations reached. Proceeding to test set...')
    else:
        dirs = [int(it[-1]) for it in os.listdir() if 'it' in it]
        iteration = max(dirs)
    print('====== Testing ======'.format(iteration))
    mkdir('testing')

    shcopy('it{}/data.hdf5'.format(iteration), 'testing/data.hdf5')
    shcopytree('it{}/nxc'.format(iteration), 'testing/nxc')
    os.chdir('testing')
    mkdir('workdir')

    engine_kwargs = {'nxc': '../../nxc'}
    engine_kwargs.update(pre.get('engine_kwargs', {}))
    driver(
        read('../testing.traj', ':'),
        pre['preprocessor'].get('application', 'siesta'),
        workdir='workdir',
        nworkers=pre.get('n_workers', 1),
        kwargs=engine_kwargs)
    add_data_driver(
        hdf5='data.hdf5',
        system='system',
        method='testing/ref',
        add=['energy'],
        traj='../testing.traj',
        override=True,
        zero=E0)
    add_data_driver(
        hdf5='data.hdf5',
        system='system',
        method='testing/nxc',
        add=['energy'],
        traj='workdir/results.traj',
        override=True,
        zero=E0)

    statistics_test = eval_driver(hdf5=['data.hdf5', 'system/testing/nxc', 'system/testing/ref'])

    open('statistics_test', 'w').write(json.dumps(statistics_test))

    os.chdir('..')
    shcopytree('testing/nxc'.format(iteration), 'final_model')
    stats = create_report()
    stats.to_csv('final_model/stats.csv', index=True, index_label='Run')
    open('final_model/stats.html', 'w').write(stats.to_html())


def fit_driver(preprocessor,
               hyper,
               hdf5=None,
               sets='',
               sample='',
               cutoff=0.0,
               model='',
               ensemble=False,
               hyperopt=False,
               b=-1,
               target_scale=1):
    """ Fits a NXCPipeline to the provided data
    """
    inputfile = hyper

    if sets != '':
        hdf5 = parse_sets_input(sets)
    else:
        hdf5 = hdf5

    inp = json.loads(open(inputfile, 'r').read())
    pre = json.loads(open(preprocessor, 'r').read())

    apply_to = []
    for pidx, path in enumerate(hdf5[1]):
        if path[0] == '*':
            apply_to.append(pidx)
            hdf5[1][pidx] = path[1:]

    grid_cv = get_grid_cv(hdf5, preprocessor, inputfile, spec_agnostic=pre['preprocessor'].get('spec_agnostic', False))

    new_model = grid_cv.estimator
    param_grid = grid_cv.param_grid
    param_grid = {key: param_grid[key][0] for key in param_grid}
    new_model.set_params(**param_grid)

    if model:
        if ensemble:
            new_model.steps[-1][1].steps[2:-1] = xc.ml.network.load_pipeline(model).steps[:-1]
        else:
            new_model.steps[-1][1].steps[2:] = xc.ml.network.load_pipeline(model).steps

    datafile = h5py.File(hdf5[0], 'r')
    basis_key = basis_to_hash(pre['preprocessor'])
    data = load_sets(datafile, hdf5[1], hdf5[2], basis_key, cutoff)

    if model:
        if not new_model.steps[-1][1].steps[-1][1].fitted:
            pipeline = Pipeline(new_model.steps[-1][1].steps[:-1])
        else:
            pipeline = new_model
        for set in apply_to:
            selection = (data[:, 0] == set)
            prediction = pipeline.predict(data)[set]
            print('Dataset {} old STD: {}'.format(set, np.std(data[selection][:, -1])))
            data[selection, -1] += prediction
            print('Dataset {} new STD: {}'.format(set, np.std(data[selection][:, -1])))

    if sample != '':
        sample = np.load(sample)
        data = data[sample]
        print("Using sample of size {}".format(len(sample)))

    class FilePipeline():
        def __init__(self, path):
            self.path = path
            self.pipeline = pickle.load(open(self.path, 'rb'))

        def predict(self, X, *args, **kwargs):
            return self.pipeline.predict(X, *args, **kwargs)

        def fit(self, X, y=None):
            print('Fitting...')
            return self

        def transform(self, X, y=None, **fit_params):
            return self.pipeline.transform(X, **fit_params)

        def fit_transform(self, X, y=None):
            return self.pipeline.transform(X)

    np.random.shuffle(data)
    if hyperopt:
        if model:
            if (new_model.steps[-1][1].steps[-2][1], NumpyNetworkEstimator) or ensemble:
                new_param_grid = {key[len('ml__'):]: value for key,value in  grid_cv.param_grid.items()\
                    if 'ml__estimator' in key}

                pickle.dump(Pipeline(new_model.steps[-1][1].steps[2:-1]), open('.tmp.pckl', 'wb'))
                estimator = GridSearchCV(
                    Pipeline(new_model.steps[-1][1].steps[:2] + [('file_pipe', FilePipeline('.tmp.pckl')),
                                                                 ('estimator', new_model.steps[-1][1].steps[-1][1])]),
                    new_param_grid,
                    cv=inp.get('cv', 2))

                new_model.steps[-1][1].steps = new_model.steps[-1][1].steps[:-1]
                do_concat = True
            else:
                raise Exception('Cannot optimize hyperparameters for fitted model')
        else:
            estimator = grid_cv
            do_concat = False
    else:
        estimator = new_model

    # Test if every fold contains at least one sample from every dataset
    passed_test = False
    n_sets = len(np.unique(data[:, 0]))
    n_splits = inp.get('cv', 2)
    while (not passed_test):
        groups = data[:int(np.floor(len(data) / n_splits) * n_splits)].reshape(n_splits,
                                                                               int(np.floor(len(data) / n_splits)),
                                                                               data.shape[-1])[:, :, 0]
        n_unique = np.array([len(np.unique(g)) for g in groups])
        if not np.all(n_unique == n_sets):
            print(np.all(n_unique == n_sets))
            np.random.shuffle(data)
        else:
            passed_test = True




    if not b == -1:
        print('Setting weight decay to b = {}'.format(b))
        estimator.steps[-1][1].steps[-1][-1].b = b

    if not target_scale == 1:
        print('Scaling targets by factor: {}'.format(target_scale))
    real_targets = np.array(data[:, -1]).real.flatten()
    data[:, -1] = data[:, -1] * target_scale

    estimator.fit(data)
    set_selection = (data[:, 0] == 0)
    dev = estimator.predict(data)[0].flatten() - real_targets
    dev0 = np.abs(dev - np.mean(dev))
    results = {
        'mean deviation': np.mean(dev).round(4),
        'rmse': np.std(dev).round(4),
        'mae': np.mean(dev0).round(4),
        'max': np.max(dev0).round(4)
    }

    if hyperopt:
        bp = estimator.best_params_
        bp = {key[len('ml__'):]: bp[key] for key in bp}
        open('best_params.json', 'w').write(json.dumps({'hyperparameters': bp}, indent=4))
        pd.DataFrame(estimator.cv_results_).to_csv('cv_results.csv')
        best_params_ = estimator.best_params_
        if do_concat:
            os.remove('.tmp.pckl')
            new_model.steps[-1][1].steps.append(estimator.best_estimator_.steps[-1])
            new_model.steps[-1][1].start_at(2).save('best_model', True)
        else:
            best_estimator = estimator.best_estimator_.steps[-1][1].start_at(2)
            best_estimator.save('best_model', True)
    else:
        estimator = estimator.steps[-1][1]
        estimator.start_at(2).save('best_model', True)
    return results


def chain_driver(hyper, model, dest='chained_estimator'):

    inputfile = hyper

    inp = json.loads(open(inputfile, 'r').read())

    param_grid = {
        key[len('estimator__'):]: value
        for key, value in inp['hyperparameters'].items() if 'estimator' in key
    }
    new_model = NetworkWrapper(**param_grid)

    old_model = xc.ml.network.load_pipeline(model)

    old_estimator = old_model.steps[-1][1]
    if not isinstance(old_estimator, xc.ml.network.NumpyNetworkEstimator):
        raise Exception('Currently only supported if saved model is NumpyNetworkEstimator,\
        but is ', type(old_estimator))

    old_model.steps[-1] = ('frozen_estimator', old_model.steps[-1][1])
    old_model.steps += [('estimator', new_model)]
    old_model.save(dest, True)


def eval_driver(hdf5, model='', plot=False, savefig='', cutoff=0.0, predict=False, dest='prediction'):
    """ Evaluate fitted NXCPipeline on dataset and report statistics
    """
    hdf5 = hdf5

    if predict:
        hdf5.append(hdf5[1])
        cutoff = 0
    else:
        cutoff = cutoff
    datafile = h5py.File(hdf5[0], 'r')

    if not model == '':
        model = xc.NeuralXC(model)._pipeline
        basis = model.get_basis_instructions()
        basis_key = basis_to_hash(basis)
    else:
        basis_key = ''

    data = load_sets(datafile, hdf5[1], hdf5[2], basis_key, cutoff)
    results = {}
    if not model == '':
        symmetrizer_instructions = model.get_symmetrize_instructions()
        symmetrizer_instructions.update({'basis': basis})
        species = [''.join(find_attr_in_tree(datafile, hdf5[1], 'species'))]
        spec_group = SpeciesGrouper(basis, species)
        symmetrizer = symmetrizer_factory(symmetrizer_instructions)
        print('Symmetrizer instructions', symmetrizer_instructions)
        pipeline = NXCPipeline(
            [('spec_group', spec_group), ('symmetrizer', symmetrizer)] + model.steps,
            basis_instructions=basis,
            symmetrize_instructions=symmetrizer_instructions)

        targets = data[:, -1].real
        predictions = pipeline.predict(data)[0]
        if predict:
            np.save(dest, predictions)
            return 0
        dev = (predictions.flatten() - targets.flatten())
    else:
        if predict:
            raise Exception('Must provide a model to make predictions')
        dev = data[:, -1].real
        # predictions = load_sets(datafile, hdf5[1], hdf5[1], basis_key, cutoff)[:,-1].flatten()
        # targets = load_sets(datafile, hdf5[2], hdf5[2], basis_key, cutoff)[:,-1].flatten()

        predictions = datafile[hdf5[1] + '/energy'][:]
        targets = datafile[hdf5[2] + '/energy'][:]
        try:
            force_base = datafile[hdf5[1] + '/forces'][:]
            force_ref = datafile[hdf5[2] + '/forces'][:]
            force_results = {
                'force_mae': np.mean(np.abs(force_ref - force_base)),
                'force_std': np.std(force_ref - force_base),
                'force_max': np.max(force_ref - force_base)
            }
            results.update(force_results)
        except Exception:
            pass

    dev0 = np.abs(dev - np.mean(dev))
    results.update({
        'mean deviation': np.mean(dev).round(4),
        'rmse': np.std(dev).round(4),
        'mae': np.mean(dev0).round(4),
        'max': np.max(dev0).round(4)
    })
    pprint(results)
    if plot:
        if model == '':
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.hist(dev.flatten())
            plt.xlabel('Target energies [eV]')
            plt.subplot(2, 1, 2)
        targets -= np.mean(targets)
        predictions -= np.mean(predictions)
        maxlim = np.max([np.max(targets), np.max(predictions)])
        minlim = np.min([np.max(targets), np.min(predictions)])
        plt.plot(targets.flatten(), predictions.flatten(), ls='', marker='.')
        plt.plot([minlim, maxlim], [minlim, maxlim], ls='-', marker='', color='grey')
        plt.xlabel('$E_{ref}[eV]$')
        plt.ylabel('$E_{pred}[eV]$')
        plt.show()
    return results


def ensemble_driver(models, operation='sum', estonly=False, dest='stacked_ensemble'):

    all_pipelines = []
    for model_path in models:
        all_pipelines.append(xc.ml.network.load_pipeline(model_path))

    if estonly:
        all_networks = [pipeline.steps[-1][1] for pipeline in all_pipelines]
        ensemble = StackedEstimator(all_networks, operation=operation)
        pipeline = all_pipelines[0]
        pipeline.steps[-1] = ('estimator', ensemble)
    else:
        ensemble = StackedEstimator(all_pipelines, operation=operation)
        pipeline = xc.ml.network.load_pipeline(model_path)
        pipeline.steps = [('estimator', ensemble)]
    pipeline.save(dest, override=True)

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
from neuralxc.ml.ensemble import StackedEstimator,ChainedEstimator
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
from .other import *
os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'

def convert_tf(args):
    """ Converts the tensorflow estimator inside a NXCPipeline to a simple
    numpy base estimator"""

    nxc_tf = xc.NeuralXC(args.tf)
    pipeline = nxc_tf._pipeline

    #Needs to do a fake run to build the tensorflow graph
    unitcell = np.eye(3)*20
    grid = [40]*3
    rho = np.zeros(grid)

    species = [key for key in pipeline.get_basis_instructions() if len(key) == 1]
    positions = np.zeros([len(species),3])

    nxc_tf.initialize(unitcell, grid, positions, species)
    # nxc_tf.get_V(rho)
    C = nxc_tf.projector.get_basis_rep(rho, positions, species)
    D = nxc_tf.symmetrizer.get_symmetrized(C)
    nxc_tf._pipeline.predict(D)
    nxc_tf._pipeline.save(args.np, True, True)

def merge_driver(args):
    """ Converts the tensorflow estimator inside a NXCPipeline to a simple
    numpy base estimator"""

    nxc_tf = xc.NeuralXC(args.chained)
    pipeline = nxc_tf._pipeline

    label, estimator = pipeline.steps[-1]
    _ , npestimator = pipeline.steps[-2]

    if not isinstance(npestimator, NumpyNetworkEstimator):
        raise Exception('Something went wrong. Second to last pipeline element'\
        +' must be NumpyNetworkEstimator')

    if not isinstance(estimator, NumpyNetworkEstimator):
        if not isinstance(estimator, NetworkWrapper):
            raise Exception('Something went wrong. Last pipeline element'\
            +' must be an estimator')
        else:
            convert_tf(namedtuple('tuplename',('tf','np'))(args.chained, args.merged))
            args.chained = args.merged

            nxc_tf = xc.NeuralXC(args.chained)
            pipeline = nxc_tf._pipeline

            label, estimator = pipeline.steps[-1]
            _ , npestimator = pipeline.steps[-2]

    if not npestimator.trunc:
            npestimator = npestimator.trunc_after(-1)

    pipeline.steps[-2] = (label, ChainedEstimator([npestimator, estimator]).merge())
    pipeline.steps = pipeline.steps[:-1]
    nxc_tf._pipeline.save(args.merged, True, True)

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

def workflow_driver(args):
    statistics_sc = {'mae': 1000}
    if args.sets:
        args.sets = os.path.abspath(args.sets)

    if args.model0:
        args.model0 = os.path.abspath(args.model0)
        open('siesta.fdf','a').write('\nNeuralXC ../../nxc\n')
        ensemble = True
    else:
        ensemble = False

    if args.nozero:
        E0 = 0
    else:
        E0 = None

    if args.hotstart == 0:
        if args.data:
            mkdir('it0')
            shcopy(args.data, 'it0/data.hdf5')
            shcopy(args.preprocessor, 'it0/pre.json')
            shcopy(args.config, 'it0/hyper.json')
            os.chdir('it0')
            open('sets.inp','w').write('data.hdf5 \n system/base \t system/ref')
            if args.sets:
                open('sets.inp','a').write('\n' + open(args.sets,'r').read())
            statistics_sc = \
            eval_driver(SN(model = '',hdf5=['data.hdf5','system/base',
                    'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=False))

            open('statistics_sc','w').write(json.dumps(statistics_sc))
            statistics_fit = fit_driver(SN(preprocessor='pre.json',config='hyper.json',mask=False, sample='',
                        cutoff=0.0, model = args.model0,ensemble=ensemble,
                        sets='sets.inp', hyperopt=True))
            open('statistics_fit','w').write(json.dumps(statistics_fit))
            convert_tf(SN(tf='best_model',np='merged_new'))
            os.chdir('../')
        else:
            iteration = 0
            print('====== Iteration {} ======'.format(iteration))
            if ensemble:
                shcopytree(args.model0,'it0/nxc')
            mkdir('it{}'.format(iteration))
            shcopy(args.preprocessor, 'it{}/pre.json'.format(iteration))
            shcopy(args.config, 'it{}/hyper.json'.format(iteration))
            os.chdir('it{}'.format(iteration))
            open('sets.inp','w').write('data.hdf5 \n system/it{} \t system/ref'.format(iteration))
            if args.sets:
                open('sets.inp','a').write('\n' + open(args.sets,'r').read())
            mkdir('workdir')
            subprocess.Popen(open('../' + args.engine,'r').read().strip() + ' ../sampled.traj', shell=True).wait()
            pre_driver(SN(preprocessor='pre.json',dest='data.hdf5/system/it{}'.format(iteration),
                            mask = False, xyz=False))
            add_data_driver(SN(hdf5='data.hdf5',system='system',method='it0',add=['energy'],
                            traj ='workdir/results.traj', density='',override=True, slice=':', zero=E0))
            add_data_driver(SN(hdf5='data.hdf5',system='system',method='ref',add=['energy'],
                            traj ='../sampled.traj', density='',override=True, slice=':', zero = E0))
            statistics_sc = \
            eval_driver(SN(model = '',hdf5=['data.hdf5','system/it{}'.format(iteration),
                    'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=False))

            open('statistics_sc','w').write(json.dumps(statistics_sc))
            statistics_fit = fit_driver(SN(preprocessor='pre.json',config='hyper.json',mask=False, sample='',
                        cutoff=0.0, model = args.model0,ensemble=ensemble,
                        sets='sets.inp', hyperopt=True))

            open('statistics_fit','w').write(json.dumps(statistics_fit))
            convert_tf(SN(tf='best_model',np='merged_new'))

            os.chdir('../')
        args.hotstart += 1
    open('siesta.fdf','a').write('\nNeuralXC ../../nxc\n')
    if not args.hotstart == -1:
        for iteration in range(args.hotstart, args.maxit +1):
            print('====== Iteration {} ======'.format(iteration))
            mkdir('it{}'.format(iteration))
            shcopy('it{}/data.hdf5'.format(iteration - 1),'it{}/data.hdf5'.format(iteration))
            shcopy(args.preprocessor, 'it{}/pre.json'.format(iteration))
            if args.config2:
                shcopy(args.config2, 'it{}/hyper.json'.format(iteration))
            else:
                shcopy(args.config, 'it{}/hyper.json'.format(iteration))
            shcopytree('it{}/merged_new'.format(iteration - 1),'it{}/merged'.format(iteration))
            os.chdir('it{}'.format(iteration))
            if ensemble:
                ensemble_driver(SN(operation='sum',dest='nxc',models=[args.model0, 'merged']))
            else:
                shcopytree('merged','nxc')

            open('sets.inp','w').write('data.hdf5 \n *system/it{} \t system/ref'.format(iteration))
            if args.sets:
                open('sets.inp','a').write('\n' + open(args.sets,'r').read())
            mkdir('workdir')
            subprocess.Popen(open('../' + args.engine,'r').read().strip() + ' ../sampled.traj', shell=True).wait()
            pre_driver(SN(preprocessor='pre.json',dest='data.hdf5/system/it{}'.format(iteration),
                            mask = False, xyz=False))

            add_data_driver(SN(hdf5='data.hdf5',system='system',method='it{}'.format(iteration),add=['energy'],
                            traj ='workdir/results.traj', density='',override=True, slice=':', zero = E0))
            old_statistics = dict(statistics_sc)
            statistics_sc = \
            eval_driver(SN(model = '',hdf5=['data.hdf5','system/it{}'.format(iteration),
                    'system/ref'],plot=False,savefig=False,cutoff=0.0,predict=False))

            open('statistics_sc','w').write(json.dumps(statistics_sc))

            if old_statistics['mae'] - statistics_sc['mae'] < args.tol:
                if old_statistics['mae'] - statistics_sc['mae'] < 0:
                    print('Self-consistent error increased in this iteration: dMAE = {} eV'.format(old_statistics['mae'] - statistics_sc['mae']))
                    iteration -= 1
                    os.chdir('../')
                else:
                    print('Iterative training converged: dMAE = {} eV'.format(old_statistics['mae'] - statistics_sc['mae']))
                    os.chdir('../')
                break

            chain_driver(SN(config='hyper.json', model ='merged',dest ='chained'))
            statistics_fit = fit_driver(SN(preprocessor='pre.json',config='hyper.json',mask=False, sample='',
                        cutoff=0.0, model = 'chained',ensemble=False,
                        sets='sets.inp', hyperopt=True))

            open('statistics_fit','w').write(json.dumps(statistics_fit))
            if statistics_fit['mae'] > statistics_sc['mae'] + args.tol:
                print('Stopping iterative training because fitting error is larger than self-consistent error')
                os.chdir('../')
                break
            merge_driver(SN(chained='best_model',merged='merged_new'))

            os.chdir('../')
        else:
            print('Maximum number of iterations reached. Proceeding to test set...')
    else:
        dirs = [int(it[-1]) for it  in os.listdir() if 'it' in it]
        iteration = max(dirs)
    print('====== Testing ======'.format(iteration))
    mkdir('testing')

    shcopy('it{}/data.hdf5'.format(iteration),'testing/data.hdf5')
    shcopytree('it{}/nxc'.format(iteration),'testing/nxc')
    os.chdir('testing')
    mkdir('workdir')
    subprocess.Popen(open('../' + args.engine,'r').read().strip() + ' ../testing.traj', shell=True).wait()

    add_data_driver(SN(hdf5='data.hdf5',system='system',method='testing/ref',add=['energy'],
                    traj ='../testing.traj', density='',override=True, slice=':', zero = E0))
    add_data_driver(SN(hdf5='data.hdf5',system='system',method='testing/nxc',add=['energy'],
                    traj ='workdir/results.traj', density='',override=True, slice=':', zero = E0))

    statistics_test = eval_driver(SN(model = '',hdf5=['data.hdf5','system/testing/nxc',
            'system/testing/ref'],plot=False,savefig=False,cutoff=0.0,predict=False))
    open('statistics_test','w').write(json.dumps(statistics_test))

def fit_driver(args):
    """ Fits a NXCPipeline to the provided data
    """
    inputfile = args.config
    preprocessor = args.preprocessor

    if args.sets != '':
        hdf5 = parse_sets_input(args.sets)
    else:
        hdf5 = args.hdf5

    mask = args.mask

    if not mask:
        inp = json.loads(open(inputfile,'r').read())
        pre = json.loads(open(preprocessor,'r').read())
    else:
        inp = {}
        pre = {}

    apply_to = []
    for pidx, path in enumerate(hdf5[1]):
        if path[0] == '*':
            apply_to.append(pidx)
            hdf5[1][pidx] = path[1:]

    grid_cv = get_grid_cv(hdf5, preprocessor, inputfile, mask)
    if mask: return 0

    new_model = grid_cv.estimator
    param_grid = grid_cv.param_grid
    param_grid = {key: param_grid[key][0] for key in param_grid}
    new_model.set_params(**param_grid)

    if args.model:
        if args.ensemble:
            new_model.steps[-1][1].steps[2:-1] =  xc.ml.network.load_pipeline(args.model).steps[:-1]
        else:
            new_model.steps[-1][1].steps[2:] =  xc.ml.network.load_pipeline(args.model).steps

    datafile = h5py.File(hdf5[0],'r')
    basis_key = basis_to_hash(pre['basis'])
    data = load_sets(datafile, hdf5[1], hdf5[2], basis_key, args.cutoff)

    if args.model:
        if new_model.steps[-1][1].steps[-1][1]._network == None:
            pipeline = Pipeline(new_model.steps[-1][1].steps[:-1])
        else:
            pipeline = new_model
        for set in apply_to:
            selection = (data[:,0] == set)
            prediction = pipeline.predict(data)[set]
            print('Dataset {} old STD: {}'.format(set, np.std(data[selection][:,-1])))
            data[selection,-1] += prediction
            print('Dataset {} new STD: {}'.format(set, np.std(data[selection][:,-1])))

    if args.sample != '':
        sample = np.load(args.sample)
        data = data[sample]
        print("Using sample of size {}".format(len(sample)))

    class FilePipeline():
        def __init__(self, path):
            self.path = path
            self.pipeline = pickle.load(open(self.path,'rb'))

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
    if args.hyperopt:
        if args.model:
            if (new_model.steps[-1][1].steps[-2][1], NumpyNetworkEstimator) or args.ensemble:
                new_param_grid = {key[len('ml__'):]: value for key,value in  grid_cv.param_grid.items()\
                    if 'ml__estimator' in key}


                pickle.dump(Pipeline(new_model.steps[-1][1].steps[2:-1]),open('.tmp.pckl','wb'))
                estimator = GridSearchCV(Pipeline(new_model.steps[-1][1].steps[:2] + [('file_pipe', FilePipeline('.tmp.pckl')),
                                                    ('estimator', new_model.steps[-1][1].steps[-1][1])]),
                                                    new_param_grid, cv=inp.get('cv',2))


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
    n_sets = len(np.unique(data[:,0]))
    n_splits = inp.get('cv',2)
    while(not passed_test):
        groups = data[:int(np.floor(len(data)/n_splits)*n_splits)].reshape(n_splits,
         int(np.floor(len(data)/n_splits)), data.shape[-1])[:,:,0]
        n_unique = np.array([len(np.unique(g)) for g in groups])
        if not np.all(n_unique == n_sets):
            print(np.all(n_unique == n_sets))
            np.random.shuffle(data)
        else:
            passed_test = True

    estimator.fit(data)
    set_selection = (data[:,0] == 0)
    dev = estimator.predict(data)[0].flatten() - data[set_selection,-1].real.flatten()
    dev0 = np.abs(dev - np.mean(dev))
    results = {'mean deviation' : np.mean(dev).round(4), 'rmse': np.std(dev).round(4),
               'mae' : np.mean(dev0).round(4),'max': np.max(dev0).round(4)}

    if args.hyperopt:
        open('best_params.json','w').write(json.dumps(estimator.best_params_, indent=4))
        pd.DataFrame(estimator.cv_results_).to_csv('cv_results.csv')
        best_params_ = estimator.best_params_
        if do_concat:
            os.remove('.tmp.pckl')
            new_model.steps[-1][1].steps.append(estimator.best_estimator_.steps[-1])
            new_model.steps[-1][1].start_at(2).save('best_model',True)
        else:
            best_estimator = estimator.best_estimator_.steps[-1][1].start_at(2)
            best_estimator.save('best_model',True)
    else:
        estimator = estimator.steps[-1][1]
        estimator.start_at(2).save('best_model',True)
    return results

def chain_driver(args):

    inputfile = args.config

    inp = json.loads(open(inputfile,'r').read())

    param_grid = {key[len('estimator__'):]:value for key,value in inp['hyperparameters'].items() if 'estimator' in key}
    new_model = NetworkWrapper(**param_grid)

    old_model = xc.ml.network.load_pipeline(args.model)

    old_estimator = old_model.steps[-1][1]
    if not isinstance(old_estimator, xc.ml.network.NumpyNetworkEstimator):
        raise Exception('Currently only supported if saved model is NumpyNetworkEstimator,\
        but is ', type(old_estimator))

    old_model.steps[-1] = ('frozen_estimator', old_model.steps[-1][1])
    old_model.steps += [('estimator', new_model)]
    old_model.save(args.dest, True)


def eval_driver(args):
    """ Evaluate fitted NXCPipeline on dataset and report statistics
    """
    hdf5 = args.hdf5

    if args.predict:
        hdf5.append(hdf5[1])
        cutoff = 0
    else:
        cutoff = args.cutoff
    datafile = h5py.File(hdf5[0],'r')

    if not args.model == '':
        model = xc.NeuralXC(args.model)._pipeline
        basis = model.get_basis_instructions()
        basis_key = basis_to_hash(basis)
    else:
        basis_key = ''

    data = load_sets(datafile, hdf5[1], hdf5[2], basis_key, cutoff)

    if not args.model == '':
        symmetrizer_instructions = model.get_symmetrize_instructions()
        symmetrizer_instructions.update({'basis' : basis})
        species =  [''.join(find_attr_in_tree(datafile, hdf5[1], 'species'))]
        spec_group = SpeciesGrouper(basis, species)
        symmetrizer = symmetrizer_factory(symmetrizer_instructions)
        print('Symmetrizer instructions', symmetrizer_instructions)
        pipeline = NXCPipeline([('spec_group', spec_group), ('symmetrizer', symmetrizer)
            ] + model.steps, basis_instructions=basis,
             symmetrize_instructions=symmetrizer_instructions)

        targets = data[:,-1].real
        predictions = pipeline.predict(data)[0]
        if args.predict:
            np.save(args.dest, predictions)
            return 0
        dev = (predictions.flatten() - targets.flatten())
    else:
        if args.predict:
            raise Exception('Must provide a model to make predictions')
        dev = data[:,-1].real
    dev0 = np.abs(dev - np.mean(dev))
    results = {'mean deviation' : np.mean(dev).round(4), 'rmse': np.std(dev).round(4),
               'mae' : np.mean(dev0).round(4),'max': np.max(dev0).round(4)}
    pprint(results)
    if args.plot:
        if args.model =='':
            plt.hist(dev.flatten())
            plt.xlabel('Target energies [eV]')
            plt.show()
        else:
            targets -= np.mean(targets)
            predictions -= np.mean(predictions)
            maxlim = np.max([np.max(targets),np.max(predictions)])
            minlim = np.min([np.max(targets),np.min(predictions)])
            plt.plot(targets.flatten(),predictions.flatten(),ls ='',marker='.')
            plt.plot([minlim,maxlim],[minlim,maxlim],ls ='-',marker='',color = 'grey')
            plt.xlabel('$E_{ref}[eV]$')
            plt.ylabel('$E_{pred}[eV]$')
            plt.show()
    return results

def ensemble_driver(args):

    all_pipelines = []
    for model_path in args.models:
        all_pipelines.append(xc.ml.network.load_pipeline(model_path))

    #Check for consistency
    # for pidx, pipeline in enumerate(all_pipelines):
        # for step0, step1 in zip(all_pipelines[0].steps[:-1], pipeline.steps[:-1]):
            # if not pickle.dumps(step0[1]) == pickle.dumps(step1[1]):
                # raise Exception('Parameters for {} in model {} inconsistent'.format(type(step0[1]), pidx))

    all_networks = [pipeline.steps[-1][1] for pipeline in all_pipelines]
    ensemble = StackedEstimator(all_networks, operation = args.operation)
    pipeline = all_pipelines[0]
    pipeline.steps[-1] = ('estimator', ensemble)

    pipeline.save(args.dest, override=True)

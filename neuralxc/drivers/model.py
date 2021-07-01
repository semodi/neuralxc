import json
import os
import shutil
from pprint import pprint

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read

import neuralxc as xc
from neuralxc.datastructures.hdf5 import *
from neuralxc.drivers.data import *
from neuralxc.drivers.other import *
from neuralxc.formatter import (SpeciesGrouper, make_nested_absolute)
from neuralxc.ml import NXCPipeline
from neuralxc.ml.utils import *
from neuralxc.preprocessor import driver
from neuralxc.symmetrizer import symmetrizer_factory
from neuralxc.utils import ConfigFile

__all__ = ['serialize', 'sc_driver', 'fit_driver', 'eval_driver']
os.environ['KMP_AFFINITY'] = 'none'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'


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


def shcopytreedel(src, dest):
    try:
        shutil.copytree(src, dest)
    except FileExistsError:
        shutil.rmtree(dest)
        shutil.copytree(src, dest)


def parse_sets_input(path):
    """ Reads a file containing the sets used for fitting.
        An asterisk before a group name indicates that model predictions
        should be subtracted from saved energies (relevant for self consistent
        training)
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


def pyscf_to_gaussian_basis(basis):

    for spec in basis:
        if len(spec) < 3:
            basis[spec] = {'basis': basis['basis'], 'sigma': 10}
    basis.pop('basis')
    return basis


def serialize(in_path, jit_path, as_radial):
    """ serialize/serialize torch model so that it can be used by libnxc
    """

    model = xc.ml.network.load_pipeline(in_path)
    projector_type = model.get_basis_instructions().get('projector_type', 'ortho')
    if as_radial:
        if not 'radial' in projector_type:
            projector_type += '_radial'

            if projector_type == 'pyscf_radial' or \
             (projector_type == 'ortho_radial' and\
              model.get_basis_instructions().get('application','') == 'pyscf'):
                projector_type = 'gaussian_radial'
                model.basis_instructions = pyscf_to_gaussian_basis(model.basis_instructions)

            model.basis_instructions.update({'projector_type': projector_type})
    else:
        if projector_type[-len('_radial'):] == '_radial':
            projector_type = projector_type[:-len('_radial')]
            model.basis_instructions.update({'projector_type': projector_type})
    xc.ml.pipeline.serialize_pipeline(model, jit_path, override=True)
    if model.get_basis_instructions().get('spec_agnostic', 'False'):
        with open(jit_path + '/AGN', 'w') as file:
            file.write('# This model is species agnostic')

    if os.path.exists('.tmp.np'):
        shutil.rmtree('.tmp.np')
    print('Success!')


def sc_driver(xyz,
              preprocessor,
              hyper,
              data='',
              maxit=5,
              tol=0.0005,
              sets='',
              nozero=False,
              model0='',
              hyperopt=False,
              keep_itdata=False):

    xyz = os.path.abspath(xyz)
    pre = make_nested_absolute(ConfigFile(preprocessor))
    engine_kwargs = pre.get('engine_kwargs', {})
    if sets:
        sets = os.path.abspath(sets)

    # ============ Start from pre-trained model ================
    # serialize it for self-consistent deployment but keep original version
    # to continue training it
    model0_orig = ''
    if model0:
        if model0 == 'model0.jit':
            raise Exception('Please choose a different name/path for model0 as it' +\
            ' model0.jit would be overwritten by this routine')
        serialize(in_path=model0, jit_path='model0.jit', as_radial=False)

        model0_orig = model0
        model0_orig = os.path.abspath(model0_orig)

        model0 = 'model0.jit'
        model0 = os.path.abspath(model0)

        engine_kwargs = {'nxc': model0}
        engine_kwargs.update(pre.get('engine_kwargs', {}))

    # If not nozero, automatically aligns energies between reference and
    # baseline data by removing mean deviation
    if nozero:
        E0 = 0
    else:
        E0 = None

    #============= Iteration 0 =================
    # Initial self-consistent calculation either with model0 or baseline method only
    # if not neuralxc model provided. Hyperparameter optimization done in first fit
    # and are kept for subsequent iterations.
    print('====== Iteration 0 ======')
    mkdir('sc')
    shcopy(preprocessor, 'sc/pre.json')
    shcopy(hyper, 'sc/hyper.json')
    os.chdir('sc')

    iteration = 0
    if model0:
        open('sets.inp', 'w').write('data.hdf5 \n *system/it{} \t system/ref'.format(iteration))
    else:
        open('sets.inp', 'w').write('data.hdf5 \n system/it{} \t system/ref'.format(iteration))

    if sets:
        open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
    mkdir('workdir')
    driver(read(xyz, ':'),
           pre['preprocessor'].get('application', 'siesta'),
           workdir='workdir',
           nworkers=pre.get('n_workers', 1),
           kwargs=engine_kwargs)
    pre_driver(xyz, 'workdir', preprocessor='pre.json', dest='data.hdf5/system/it{}'.format(iteration))
    add_data_driver(hdf5='data.hdf5',
                    system='system',
                    method='it0',
                    add=['energy'],
                    traj='workdir/results.traj',
                    override=True,
                    zero=E0)
    add_data_driver(hdf5='data.hdf5', system='system', method='ref', add=['energy'], traj=xyz, override=True, zero=E0)
    statistics_sc = \
    eval_driver(hdf5=['data.hdf5','system/it{}'.format(iteration),
            'system/ref'])

    open('statistics_sc', 'w').write(json.dumps(statistics_sc))
    statistics_fit = fit_driver(preprocessor='pre.json',
                                hyper='hyper.json',
                                model=model0_orig,
                                sets='sets.inp',
                                hyperopt=hyperopt)

    open('statistics_fit', 'w').write(json.dumps(statistics_fit))

    #=================== Iterations > 0 ==============
    it_label = 1
    for it_label in range(1, maxit + 1):
        if keep_itdata:
            iteration = it_label
        else:
            iteration = 0

        print('====== Iteration {} ======'.format(it_label))
        open('sets.inp', 'w').write('data.hdf5 \n *system/it{} \t system/ref'.format(iteration))

        if sets:
            open('sets.inp', 'a').write('\n' + open(sets, 'r').read())
        mkdir('workdir')

        shcopytreedel('best_model', 'model_it{}'.format(it_label))
        serialize('model_it{}'.format(it_label), 'model_it{}.jit'.format(it_label),
                  'radial' in pre['preprocessor'].get('projector_type', 'ortho'))

        engine_kwargs = {'nxc': '../../model_it{}.jit'.format(it_label), 'skip_calculated': False}
        engine_kwargs.update(pre.get('engine_kwargs', {}))

        driver(read(xyz, ':'),
               pre['preprocessor'].get('application', 'siesta'),
               workdir='workdir',
               nworkers=pre.get('n_workers', 1),
               kwargs=engine_kwargs)

        pre_driver(xyz, 'workdir', preprocessor='pre.json', dest='data.hdf5/system/it{}'.format(iteration))

        add_data_driver(hdf5='data.hdf5',
                        system='system',
                        method='it{}'.format(iteration),
                        add=['energy'],
                        traj='workdir/results.traj',
                        override=True,
                        zero=E0)
        statistics_sc = \
        eval_driver(hdf5=['data.hdf5','system/it{}'.format(iteration),
                'system/ref'])

        open('statistics_sc', 'a').write('\n' + json.dumps(statistics_sc))
        open('model_it{}/statistics_sc'.format(it_label), 'w').write('\n' + json.dumps(statistics_sc))
        statistics_fit = fit_driver(preprocessor='pre.json', hyper='hyper.json', model='best_model', sets='sets.inp')
        open('statistics_fit', 'a').write('\n' + json.dumps(statistics_fit))

    os.chdir('..')
    print('====== Testing ======')
    testfile = ''
    if os.path.isfile('testing.xyz'):
        testfile = '../testing.xyz'
    if os.path.isfile('testing.traj'):
        testfile = '../testing.traj'

    if testfile:
        mkdir('testing')

        shcopy('sc/data.hdf5'.format(iteration), 'testing/data.hdf5')
        shcopytree('sc/model_it{}.jit'.format(it_label), 'testing/nxc.jit')
        shcopytree('sc/model_it{}'.format(it_label), 'final_model/')
        shcopytree('sc/model_it{}.jit'.format(it_label), 'final_model.jit/')
        os.chdir('testing')
        mkdir('workdir')
        engine_kwargs = {'nxc': '../../nxc.jit'}
        engine_kwargs.update(pre.get('engine_kwargs', {}))
        driver(read(testfile, ':'),
               pre['preprocessor'].get('application', 'siesta'),
               workdir='workdir',
               nworkers=pre.get('n_workers', 1),
               kwargs=engine_kwargs)
        add_data_driver(hdf5='data.hdf5',
                        system='system',
                        method='testing/ref',
                        add=['energy'],
                        traj=testfile,
                        override=True,
                        zero=E0)
        add_data_driver(hdf5='data.hdf5',
                        system='system',
                        method='testing/nxc',
                        add=['energy'],
                        traj='workdir/results.traj',
                        override=True,
                        zero=E0)

        statistics_test = eval_driver(hdf5=['data.hdf5', 'system/testing/nxc', 'system/testing/ref'])
        open('statistics_test', 'w').write(json.dumps(statistics_test))
        os.chdir('..')
    else:
        print('testing.traj or testing.xyz not found.')


def fit_driver(preprocessor, hyper, hdf5=None, sets='', sample='', cutoff=0.0, model='', hyperopt=False):
    """ Fits a NXCPipeline to the provided data
    """
    inputfile = hyper
    if sets != '':
        hdf5 = parse_sets_input(sets)

    pre = make_nested_absolute(ConfigFile(preprocessor))
    basis_key = basis_to_hash(pre['preprocessor'])
    if 'gaussian' in pre['preprocessor'].get('projector_type','ortho')\
        and pre['preprocessor'].get('spec_agnostic',False):
        pre['preprocessor'].update(get_real_basis(None, pre['preprocessor']['X']['basis'], True))

    print(pre['preprocessor'])

    # A * in hdf5 (if sets != '') indicates that the predictions of a pretrained
    # model should be subtracted from the stored baseline energies
    # This is relevant for self-consistent training
    apply_to = []
    for pidx, path in enumerate(hdf5[1]):
        if path[0] == '*':
            apply_to.append(pidx)
            hdf5[1][pidx] = path[1:]

    grid_cv = get_grid_cv(hdf5, pre, inputfile, spec_agnostic=pre['preprocessor'].get('spec_agnostic', False))

    new_model = grid_cv.estimator
    param_grid = grid_cv.param_grid
    param_grid = {key: param_grid[key][0] for key in param_grid}
    new_model.set_params(**param_grid)

    if model:
        hyperopt = False
        new_model.steps[-1][1].steps[2:] = xc.ml.network.load_pipeline(model).steps

    datafile = h5py.File(hdf5[0], 'r')
    data = load_sets(datafile, hdf5[1], hdf5[2], basis_key, cutoff)

    if model:
        for set in apply_to:
            selection = (data[:, 0] == set)
            prediction = new_model.predict(data)[set][:, 0]
            print('Dataset {} old STD: {}'.format(set, np.std(data[selection][:, -1])))
            data[selection, -1] += prediction
            print('Dataset {} new STD: {}'.format(set, np.std(data[selection][:, -1])))

    if sample != '':
        sample = np.load(sample)
        data = data[sample]
        print("Using sample of size {}".format(len(sample)))

    np.random.shuffle(data)
    if hyperopt:
        estimator = grid_cv
    else:
        estimator = new_model

    real_targets = np.array(data[:, -1]).real.flatten()

    estimator.fit(data)

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
        best_estimator = estimator.best_estimator_.steps[-1][1].start_at(2)
        best_estimator.save('best_model', True)
    else:
        estimator = estimator.steps[-1][1]
        estimator.start_at(2).save('best_model', True)
    return results


def eval_driver(hdf5,
                model='',
                plot=False,
                savefig='',
                cutoff=0.0,
                predict=False,
                dest='prediction',
                sample='',
                invert_sample=False,
                keep_mean=False,
                hashkey=''):
    """ Evaluate fitted NXCPipeline on dataset and report statistics
    """

    if predict:
        hdf5.append(hdf5[1])
        cutoff = 0

    datafile = h5py.File(hdf5[0], 'r')

    if not model == '':
        model = xc.ml.network.load_pipeline(model)
        basis = model.get_basis_instructions()
        if not hashkey:
            basis_key = basis_to_hash(basis)
        else:
            basis_key = hashkey
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
        pipeline = NXCPipeline([('spec_group', spec_group), ('symmetrizer', symmetrizer)] + model.steps,
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

    if sample:
        sample = np.load(sample)
        if invert_sample:
            sample = np.array([f for f in np.arange(len(dev)) if not f in sample])
        dev = dev[sample]
        targets = targets[sample]
        predictions = predictions[sample]
    if keep_mean:
        dev0 = np.abs(dev)
    else:
        dev0 = np.abs(dev - np.mean(dev))
    results.update({
        'mean deviation': np.mean(dev).round(5),
        'rmse': np.sqrt(np.mean(dev**2)).round(5),
        'mae': np.mean(dev0).round(5),
        'max': np.max(dev0).round(5)
    })
    pprint(results)
    if plot:
        if model == '':
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.hist(dev.flatten(), bins=np.max([10, int(len(dev.flatten()) / 100)]))
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

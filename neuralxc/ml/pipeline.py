"""
pipeline.py
Implements NXCPipeline, an extension of scikit-learn Pipelines that is fully
differentiable.
Contains routines to serialize functionals into TorchScript models.
"""

import json
import os
import shutil

import dill as pickle
import numpy as np
import torch
from sklearn.pipeline import Pipeline

from ..projector import DensityProjector
from ..symmetrizer import Symmetrizer

TorchModule = torch.nn.Module
# import tensorflow


def convert_torch_wrapper(func):
    def wrapped_func(X, *args, **kwargs):
        X = torch.from_numpy(X)
        Y = func(X, *args, **kwargs)
        return Y.detach().numpy()

    return wrapped_func


class NXCPipeline(Pipeline):
    def __init__(self, steps, basis_instructions, symmetrize_instructions):
        """ Class that extends the scikit-learn Pipeline by adding get_gradient
        and save methods.

        Parameters
        -----------

        steps: list
            List of Transformers with final step being an Estimator
        basis_instructions: dict
            Dictionary containing instructions for the projector.
            Example {'C':{'n' : 4, 'l' : 2, 'r_o' : 3}}
        symmetrize_instructions: dict
            Instructions for symmetrizer.
            Example {symmetrizer_type: 'trace'}
        """
        # torch.nn.Module.__init__(self)
        self.basis_instructions = basis_instructions
        self.symmetrize_instructions = symmetrize_instructions
        self.steps = steps
        super().__init__(steps)

    def get_symmetrize_instructions(self):
        return self.symmetrize_instructions

    def get_basis_instructions(self):
        return self.basis_instructions

    def start_at(self, step_idx):
        """ Return a new NXCPipeline containing a subset of steps of the
        original NXCPipeline

        Parameters
        ----------
        step_idx: int
            Use all steps following and including step with index step_idx
        """

        return NXCPipeline(self.steps[step_idx:],
                           basis_instructions=self.basis_instructions,
                           symmetrize_instructions=self.symmetrize_instructions)

    def save(self, path, override=False, npmodel=False):
        """ Save entire pipeline to disk.

        Parameters
        ----------
        path: string
            Directory in which to store pipeline
        override: bool
            If directory already exists, only save and override if this
            is set to True

        """
        if os.path.isdir(path):
            if not override:
                raise Exception('Model already exists, set override = True')
            else:
                shutil.rmtree(path)
                os.mkdir(path)
        else:
            os.mkdir(path)

        pickle.dump([self.steps, self.basis_instructions, self.symmetrize_instructions],
                    open(os.path.join(path, 'pipeline.pckl'), 'wb'))

    def to_torch(self):
        for step_idx, _ in enumerate(self.steps):
            self.steps[step_idx][1].to_torch()

    def forward(self, X):
        for steps in self.steps:
            X = steps[1].forward(X)
        return X


def load_pipeline(path):
    """ Load a NXCPipeline from the directory specified in path
    """
    steps, basis_instructions, symmetrize_instructions = \
        pickle.load(open(os.path.join(path, 'pipeline.pckl'), 'rb'))
    return NXCPipeline(steps, basis_instructions, symmetrize_instructions)


class E_predictor(TorchModule):
    def __init__(self, species, model):
        TorchModule.__init__(self)
        self.species = species
        steps = [model.symmetrizer] + [step[1] for step in model.steps]
        steps[-1] = steps[-1]._network
        self.model = torch.nn.Sequential(*steps)

    def forward(self, *args):
        C = {spec: c for spec, c in zip(self.species, args)}
        return self.model(C)


class ModuleBasis(TorchModule):
    def __init__(self, projector):
        TorchModule.__init__(self)
        self.projector = projector

    def forward(self, positions, unitcell, grid, my_box):
        # positions = torch.einsum('...i,ij->...j',positions, unitcell)
        return self.projector.forward_basis(positions, unitcell, grid, my_box)


class ModuleProject(TorchModule):
    def __init__(self, projector):
        TorchModule.__init__(self)
        self.projector = projector

    def forward(self, rho, positions, unitcell, grid, radials, angulars, my_box):
        # positions = torch.einsum('...i,ij->...j',positions, unitcell)
        return self.projector.forward_fast(rho, positions, unitcell, grid, radials, angulars, my_box)


def serialize_energy(model, C, outpath, override):

    e_models = {}
    with torch.jit.optimized_execution(should_optimize=True):
        for spec in C:
            c = torch.from_numpy(C[spec])
            epred = E_predictor(spec, model)
            e_models[spec] = torch.jit.trace(epred, c, check_trace=False)

    try:
        os.mkdir(outpath)
    except FileExistsError:
        if override:
            shutil.rmtree(outpath)
            os.mkdir(outpath)
        else:
            raise Exception('Model exists, set override = True to save at this location')

    for spec in C:
        torch.jit.save(e_models[spec], outpath + '/xc_' + spec)
        open(outpath + '/bas.json', 'w').write(json.dumps(model.basis_instructions))


def serialize_projector(projector):

    unitcell_c = np.eye(3) * 5.0
    grid_c = np.array([9, 9, 9])
    my_box = np.array([[0, 9]] * 3)
    pos_c = np.array([[0, 0, 0]])
    if 'radial' in projector._registry_name:
        rho_c = np.array([1, 2, 3])
    else:
        rho_c = np.ones(shape=grid_c)

    species = []
    for spec in projector.basis:
        if len(spec) < 3:
            species.append(spec)

    unitcell_c = torch.from_numpy(unitcell_c).double()
    grid_c = torch.from_numpy(grid_c).double()
    pos_c = torch.from_numpy(pos_c).double()
    rho_c = torch.from_numpy(rho_c).double()
    my_box = torch.from_numpy(my_box).double()
    basismod = ModuleBasis(projector)
    projector = ModuleProject(projector)

    basis_models = {}
    projector_models = {}
    with torch.jit.optimized_execution(should_optimize=True):
        for spec in species:
            basismod.projector.set_species(spec)
            basis_models[spec] = torch.jit.trace(basismod, (pos_c, unitcell_c, grid_c, my_box), check_trace=True)
            radials, angulars, box = basis_models[spec](pos_c, unitcell_c, grid_c, my_box)
            projector_models[spec] = torch.jit.trace(projector,
                                                     (rho_c, pos_c, unitcell_c, grid_c, radials, angulars, box),
                                                     check_trace=False)

    return basis_models, projector_models


def serialize_pipeline(model, outpath, override=False):

    unitcell_c = np.eye(3) * 5.0
    grid_c = np.array([9, 9, 9])
    my_box = np.array([[0, 9]] * 3)
    pos_c = np.array([[0, 0, 0]])
    basis_instructions = model.basis_instructions
    species = []
    for spec in basis_instructions:
        if len(spec) < 3:
            species.append(spec)

    model.symmetrize_instructions.update({'basis': model.basis_instructions})
    model.symmetrizer = Symmetrizer(model.symmetrize_instructions)
    try:
        projector = DensityProjector(basis_instructions=basis_instructions, unitcell=unitcell_c, grid=grid_c)
    except TypeError:
        try:
            rho_c = np.array([1, 2, 3])
            projector = DensityProjector(basis_instructions=basis_instructions,
                                         grid_coords=unitcell_c,
                                         grid_weights=grid_c)
            model.symmetrize_instructions.update(projector.symmetrize_instructions)
            model.symmetrizer = Symmetrizer(model.symmetrize_instructions)
        except TypeError:
            C = {}
            for spec in species:
                n = basis_instructions[spec]['n']
                l = basis_instructions[spec]['l']
                C[spec] = np.ones([1, n * l**2])
            serialize_energy(model, C, outpath, override)
            return 0

    unitcell_c = torch.from_numpy(unitcell_c).double()
    grid_c = torch.from_numpy(grid_c).double()
    pos_c = torch.from_numpy(pos_c).double()
    rho_c = torch.from_numpy(rho_c).double()
    my_box = torch.from_numpy(my_box).double()
    basis_models, projector_models = serialize_projector(projector)
    e_models = {}
    with torch.jit.optimized_execution(should_optimize=True):
        for spec in species:
            radials, angulars, box = basis_models[spec](pos_c, unitcell_c, grid_c, my_box)
            C = projector_models[spec](rho_c, pos_c, unitcell_c, grid_c, radials, angulars, box).unsqueeze(0)
            epred = E_predictor(spec, model)
            e_models[spec] = torch.jit.trace(epred, C, check_trace=False)

    try:
        os.mkdir(outpath)
    except FileExistsError:
        if override:
            shutil.rmtree(outpath)
            os.mkdir(outpath)
        else:
            raise Exception('Model exists, set override = True to save at this location')

    for spec in species:
        torch.jit.save(basis_models[spec], outpath + '/basis_' + spec)
        torch.jit.save(projector_models[spec], outpath + '/projector_' + spec)
        torch.jit.save(e_models[spec], outpath + '/xc_' + spec)
        open(outpath + '/bas.json', 'w').write(json.dumps(model.basis_instructions))

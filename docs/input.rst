.. _Input:

Configuration Files
===================

Users can fine-tune the behavior of NeuralXC through the use of configuration files.
These come in two forms, one of which contains information about the projection basis set and instructions to the engine
(the electronic structure code). The other file specifies the hyperparameters used in the ML model.
Although file names are arbitrary we will stick to our convention and refer to the former as ``config.json``
and the latter as ``hyper.json``.

config.json
------------

A  configuration file to be used together with **PySCF** could look like the following:
::
   {
       "preprocessor":
       {
           "basis": {"name": "cc-pVDZ-JKFIT"},
           "projector": "gaussian",
           "grid": "analytical"
       },
       "engine":
       {
           "application": "pyscf",
           "xc": "PBE",
           "basis" : "def2-TZVP"
       },
       "n_workers" : 1,
       "symmetrizer_type": "trace"
   }




``preprocessor`` contains all information required to perform the density projection, whereas ``engine`` captures everything concerning the SCF calculations (which application/engine to use, the XC-functional etc.). ``n_workers`` defines the number of processes to be used to conduct SCF calculations and projections in parallel.

The projection basis inside ``preprocessor`` can be provided in several different ways.

**GTO basis**

The user can either provide the PySCF internal name of a basis set

    - ``"basis": {"name": "cc-pVDZ-JKFIT"}``

or the path to a file containing the basis set definition (NWChem format is used)

    -  ``"basis": {"file": "./my_basis_set"}``

In addition, if the projection is done on a grid, the two additional parameters that control the basis set localization can be provided (see Sec. \ref{sec:basis})

    - ``"basis": {"name": "cc-pVDZ-JKFIT", "sigma": 2.0, "gamma": 1.0}``

**Polynomial basis**

For polynomial basis sets the basis needs to be specified as

    - ``"basis": {"n": 3, "l": 4, "r_o": 2.0}``

if the same basis is to be used for every element.

If different basis sets are desired for each element, they can be specified as
::
    "basis": {
        "O" : {"n": 3, "l": 4, "r_o": 2.0},
        "H" : {"n": 2, "l": 3, "r_o": 2.5},
    }

An example of a configuration file to be used together with **SIESTA** could be:
::
   {
       "preprocessor":
       {
           "basis": {"n": 3, "l": 4, "r_o": 2.0},
           "projector": "ortho",
           "grid": "euclidean"
       },
       "engine":
       {
           "application": "siesta",
           "xc": "PBE",
           "basis" : "DZP",
           "pseudoloc" : ".",
           "fdf_path" : "./my_input.fdf",
           "extension": "RHOXC"
       },
       "n_workers" : 1
   }


Compared to PySCF, there are three notable differences in the ``engine`` section. ``pseudoloc`` specifies the location where pseudopotential files are stored. ``fdf_path`` is optional and can be used to point to a SIESTA input file (``.fdf``) which is used to augment the engine options set in config.json. The .fdf file should \textbf{not} contain any system specific information such as atomic positions as these are automatically filled by NeuralXC.  ``extension`` specifies the extension of the files storing the electron density and can be switched between ``RHOXC``, ``RHO``, and ``DRHO`` (see SIESTA \cite{siesta} documentation for details).

hyper.json
----------

An example for a hyperparameter configuration file is shown below. We have added explanations of each line preceded by
\# (The json file format does not support comments, therefore these lines need to be removed before using the example).
::
 {
  "hyperparameters": {
   # Remove features with Var < 1e-10
      "var_selector__threshold": 1e-10,
   # Number of hidden nodes (same for every layer)
      "estimator__n_nodes": 4,
   # Number of hidden layers (0: linear regression)
      "estimator__n_layers": [0, 3],
   # L2 regularization strength
      "estimator__b": [0.01, 0.001],
   # Learning rate
      "estimator__alpha": 0.001,
   # Maximum number of training steps
      "estimator__max_steps": 20001,
   # Relative size of validation set to be split of training
      "estimator__valid_size": 0,
   # Minibatch size (use entire dataset if 0)
      "estimator__batch_size": 0,
   # Activation Function
      "estimator__activation": "GeLU"
  },
  # Number of folds for cross-validation
     "cv": 2
 }

Parameters can either provided as a single value, or through a list (indicated by square brackets). A hyperparameter optimization using k-fold cross validation (CF) can be performed over all values provided inside the list, using a number of folds specified in ``cv``. If multiple lists are given, the hyperparameter search is performed over the outer product of all lists. In case hyperparameter optimization is disabled, the first entry of each list is used by default.

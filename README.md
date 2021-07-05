NeuralXC
==============================
[//]: # (Badges)
![Python Unit Testing](https://github.com/semodi/neuralxc/actions/workflows/unittest.yml/badge.svg)
[![codecov](https://codecov.io/gh/semodi/neuralxc/branch/master/graph/badge.svg)](https://codecov.io/gh/semodi/neuralxc/branch/master)
[![DOI](https://zenodo.org/badge/175675755.svg)](https://zenodo.org/badge/latestdoi/175675755)
[![Documentation Status](https://readthedocs.org/projects/neuralxc/badge/?version=latest)](https://neuralxc.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/semodi/neuralxc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/semodi/neuralxc/context:python)

<img src="https://github.com/semodi/neuralxc/blob/master/neuralxc.png" width="500" height="280" />

Implementation of a machine learned density functional as presented [here](https://www.nature.com/articles/s41467-020-17265-7)
This project only includes routines to fit and test neural network based density functionals. To actually use these functionals within electronic structure codes please refer to [**Libnxc**](https://github.com/semodi/libnxc/)

### Installation

To install NeuralXC using pip, navigate into the root directory of the repository and run
```
sh install.sh
```
So far, NeuralXC has only been tested on Linux and Mac OS X.

To check the integrity of your installation, you can run unit tests with
```
pytest -v
```
in the root directory.

### Libnxc and pylibnxc

The new version of NeuralXC only implements the neural network architecture along with routines to **train** and test functionals. As neural networks are
trained self-consistently, an electronic structure code to drive these calculations is needed. For this purpose, we have developed [Libnxc](https://github.com/semodi/libnxc), which allows for easy interfacing with electronic structure codes such as SIESTA and CP2K. Its python version,
pylibnxc is installed automatically together with this package and works with PySCF out-of-the-box.

### How-to

To get accustomed with NeuralXC, we recommend that PySCF is used as the driver code.
Examples on how to train and deploy a machine learned functional can be found in [examples/example_scripts/](examples/example_scripts).
The files required for the tutorial in the following section can be found in [examples/quickstart/](examples/quickstart).

#### Model training

The new version of NeuralXC only implements the neural network architecture along with routines to train and test functionals. As neural networks are trained self-consistently, an electronic structure code to drive these calculations is needed. For this purpose, we have developed Libnxc, which allows for easy interfacing with electronic structure codes such as SIESTA and CP2K. Its python version, pylibnxc is installed automatically together with this package and works with PySCF out-of-the-box.

To get accustomed with NeuralXC, we recommend that PySCF is used as the driver code.
Examples on how to train and deploy a machine learned functional can be found in ``examples/example_scripts/``.

In this tutorial we use files contained in ``examples/quickstart/``. To begin, navigate into this directory.

To train/fit a functional a set of structures and their associated reference energies is required.
These structures need to be provided in an ASE formatted ``.xyz`` or ``.traj`` file (in this example training_structures.xyz).

Self-consistent training can then be performed by running::
```
  neuralxc sc training_structures.xyz config.json hyperparameters.json --hyperopt
```

- **config.json** contains information regarding the basis set as well as the 'driver' program (PySCF), other examples can be found in examples/inputs/ml_basis/.

- **hyperparameters.json** contains the machine learning hyperparameters, other examples can be found in examples/inputs/hyper.


A minimal input file structure would look something like this:

**config.json**
```
  {
    "preprocessor":
    {
         "basis": {
                 "file": "quickstart-basis"
         },
         "projector": "gaussian",
         "grid": "analytical",
         "extension": "chkpt"
    },
    "n_workers" : 1,
    "engine": {"xc": "PBE",
               "application": "pyscf",
               "basis" : "def2-TZVP"}

  }
```

**hyperparameters.json**
```
  {
   "hyperparameters": {
       "var_selector__threshold": 1e-10,
       "estimator__n_nodes": 4,
       "estimator__n_layers": 0,
       "estimator__b": [0, 0.1, 0.001],
       "estimator__alpha": 0.001,
       "estimator__max_steps": 20001,
       "estimator__valid_size": 0,
       "estimator__batch_size": 0,
       "estimator__activation": "GeLU"
   },
      "cv": 4
  }
```

A detailed explanation of these files is given in the documentation.

NeuralXC will train a model self-consistently on the provided structures. This means an initial model is fitted to the reference energies.
This model is then used to run self-consistent calculations on the dataset producing updated baseline energies. Another model is fitted on
the difference between the reference and updated baseline energies and self-consistent calculations are run with the new model. This is
done iteratively until the model error converges within a given tolerance. This tolerance can be set with the ``--tol`` flag, the default is 0.5 meV.

At the end of the self-consistent training process a ``final_model.jit`` is produced that can be used by Libnxc. If either ``testing.traj`` or
``testing.xyz`` is found in the work directory self-consistent calculations are run for these structures using the final model and the error
on the test set is reported. In our example, the final MAE should be below 10 meV.

#### Model deployment

The final model can then be used to perform self-consistent calculations on other systems. This can be done by utilizing Libnxc to run
standard DFT calculations while accessing NeuralXC models. However, in case testing needs to be conducted across other datasets (e.g. the structures
stores in ``more_testing.xyz``), it is easier to do so using the following command::
```
  neuralxc engine config_with_model.json more_testing.xyz
```
``config_with_model.json`` is identical to the original ``config.json`` except for instructions to use ``final_model.jit``. This command will
run self-consistent calculations for every structure contained in the xyz file while saving the resulting energies in ``results.traj``.
In order to quickly evaluate error metrics we can also use NeuralXC::
```
  neuralxc data add data.hdf5 more_testing final_model energy --traj results.traj
  neuralxc data add data.hdf5 more_testing reference energy --traj more_testing.traj
```
Will add both refernce values and the ones obtained with our NeuralXC functionals to a newly created ``data.hdf5``::
```
  neuralxc eval data.hdf5 more_testing/final_model more_testing/reference --plot
```
will print error statistics and show a correlation plot.

### Reproducibility

To reproduce the results presented in \[2\] please refer to our release v0.2 of this repository.

### Reference

If you use this code in your work, please cite it as

[1] *Dick, Sebastian, and Marivi Fernandez-Serra. "Learning from the density to correct total energy and forces in first principle simulations." The Journal of Chemical Physics 151.14 (2019): 144102.*

and


[2] *Dick, S., Fernandez-Serra, M. Machine learning accurate exchange and correlation functionals of the electronic density. Nat Commun 11, 3509 (2020). https://doi.org/10.1038/s41467-020-17265-7*

### Copyright

Copyright (c) 2019, Sebastian Dick


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.

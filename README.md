NeuralXC
==============================
[//]: # (Badges)
![Python Unit Testing](https://github.com/semodi/neuralxc/actions/workflows/unittest.yml/badge.svg)
[![codecov](https://codecov.io/gh/semodi/neuralxc/branch/master/graph/badge.svg)](https://codecov.io/gh/semodi/neuralxc/branch/master)
[![DOI](https://zenodo.org/badge/175675755.svg)](https://zenodo.org/badge/latestdoi/175675755)
[![Documentation Status](https://readthedocs.org/projects/neuralxc/badge/?version=latest)](https://neuralxc.readthedocs.io/en/latest/?badge=latest)

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

#### Model training

To train/fit a functional a set of structures and their associated reference energies is required. These structures need to be provided in an [ASE](https://wiki.fysik.dtu.dk/ase/) formatted `.xyz` or `.traj` file (in this example `training_structures.xyz`). Self-consistent training can then be performed by running

`neuralxc sc training_structures.xyz basis.json hyperparameters.json`

- `basis.json` contains information regarding the basis set as well as the 'driver' program (PySCF), examples can be found in [examples/inputs/ml_basis/](examples/inputs/ml_basis).   

- `hyperparameters.json` contains the machine learning hyperparameters, examples can be found in [examples/inputs/hyper](examples/inputs/hyper).

- For more options please refer to the documentation and `neuralxc sc --help`


#### Model deployment

After installing Libnxc and patching SIESTA (see instructions in [Libnxc manual](https://libnxc.readthedocs.io/en/latest/), the trained NeuralXC functionals can be used from within SIESTA in self-consistent calculations.
To deploy a trained model in SIESTA simply add the line `neuralxc $PATH_TO_NXC_MODEL` to your `.fdf` input file.

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

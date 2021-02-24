NeuralXC
==============================
[//]: # (Badges)
[![Build Status](https://travis-ci.org/semodi/neuralxc.svg?branch=master)](https://travis-ci.org/semodi/neuralxc)
[![codecov](https://codecov.io/gh/semodi/neuralxc/branch/master/graph/badge.svg)](https://codecov.io/gh/semodi/neuralxc/branch/master)
[![DOI](https://zenodo.org/badge/175675755.svg)](https://zenodo.org/badge/latestdoi/175675755)

<img src="https://github.com/semodi/neuralxc/blob/master/NeuralXC.png" width="700" height="450" />

Implementation of a machine learned density functional as presented [here](https://chemrxiv.org/articles/Machine_Learning_a_Highly_Accurate_Exchange_and_Correlation_Functional_of_the_Electronic_Density/9947312)


### Installation


To install NeuralXC, navigate into the root directory of the repository and run
```
sh install.sh
```
This assumes that anaconda is available, alternatively the packages listed in `install.sh` can be manually installed with pip .
So far, NeuralXC has only been tested on Linux and Mac OS X.

To check the integrity of your installation, you can run unite tests with
```
pytest -v
```
in the same directory.

### Libnxc and pylibnxc

The new version of NeuralXC only implement routines to **train** functionals. To actually use these functionals in
self-consistent electronic structure calculations, [Libnxc](https://github.com/semodi/libnxc) (for C++ and Fortran support) or pylibnxc (for Python support) is required.
pylibnxc is installed automatically by `sh install.sh` whereas Libnxc has to be downloaded and compiled manually.

### How-to

Out of the box, NeuralXC works with PySCF. This means  
Examples on how to train and deploy a machine learned functional can be found in [examples/example_scripts/](examples/example_scripts).

#### Model training

To train/fit a functional a set of structures and their associated reference energies is required. These structures need to be provided in an [ASE](https://wiki.fysik.dtu.dk/ase/) formatted `.xyz` or `.traj` file (in this example `training_structures.xyz`). Self-consistent training can be performed by running

`neuralxc sc training_structures.xyz basis.json hyperparameters.json`

- `basis.json` contains information regarding the basis set as well as the 'driver' program (SIESTA), examples can be found in [examples/inputs/ml_basis/](examples/inputs/ml_basis).   

- `hyperparameters.json` contains the machine learning hyperparameters, examples can be found in [examples/inputs/hyper](examples/inputs/hyper).

- For more options please refer to the documentation and `neuralxc sc --help`


#### Model deployment

After installing Libnxc and patching SIESTA (see instructions in [Libnxc manual](https://libnxc.readthedocs.io/en/latest/), NeuralXC can be used from within
SIESTA in self-consistent calculations.
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

NeuralXC
==============================
[//]: # (Badges)
[![Build Status](https://travis-ci.org/semodi/neuralxc.svg?branch=master)](https://travis-ci.org/semodi/neuralxc)
[![codecov](https://codecov.io/gh/semodi/neuralxc/branch/master/graph/badge.svg)](https://codecov.io/gh/semodi/neuralxc/branch/master)


<img src="https://github.com/semodi/neuralxc/blob/master/NeuralXC.png" width="700" height="450" />

Implementation of a machine learned density functional as presented [here](https://chemrxiv.org/articles/Machine_Learning_a_Highly_Accurate_Exchange_and_Correlation_Functional_of_the_Electronic_Density/9947312)


### Installation

Installation of NeuralXC requires anaconda. 
NeuralXC has only been tested on Linux and Mac OS X.
To install NeuralXC, navigate into the root directory of the repository and run 
```
sh install.sh 
```
The installation can be tested by running
```
pytest -v
``` 
in the same directory. Once all dependencies are downloaded and installed, the installation of NeuralXC should not take more than one minute.

#### Electronic structure code installation

At the moment, NeuralXC has been implemented to work with SIESTA (Spanish Initiative for Electronic Simulations with Thousands of Atoms).
So far, the only way to install the NeuralXC extension is by applying a patch that can be found in `src/siesta_patch.tar`. 

- To install please download the 4.1-b4 version of SIESTA [here](https://launchpad.net/siesta) and unpack it at a location of your preference.

- Proceed by copying [`src/siesta_patch.tar`](src) into the `Src/` directory inside your SIESTA installation.

- Unpack the patch by running `tar -xf siesta_patch.tar` (inside `Src/`) and run `sh apply_patch.sh`. This will apply the patch file and download [Forpy](https://github.com/ylikx/forpy), a library that is needed for SIESTA to access Python code. During compilation, Forpy requires the flags
```
-fno-lto `python3-config --ldflags`
```
If you are using one of SIESTA's arch.make files simply add this line to `LIBS=`.

- SIESTA can now be compiled as usual (MPI is still supported). Please refer to their manual regarding details about the installation.

### Examples

Examples on how to train and deploy a machine learned functional can be found in [examples/example_scripts/](examples/example_scripts).

### Reproducibility 

Data (raw data, input files, trained models) needed to reproduce the results presented in \[2\] can be found in [examples/](examples) 

### Reference

If you use this code in your work, please cite it as 

[1] *Dick, Sebastian, and Marivi Fernandez-Serra. "Learning from the density to correct total energy and forces in first principle simulations." The Journal of Chemical Physics 151.14 (2019): 144102.*

and


[2] *Dick, Sebastian, and Marivi Fernandez-Serra. "Machine Learning a Highly Accurate Exchange and Correlation Functional of the Electronic Density". ChemRxiv 9947312 (preprint), doi:10.26434/chemrxiv.9947312.v1*

### Copyright

Copyright (c) 2019, Sebastian Dick


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0. 
This project uses an implementation of the gradient of spherical harmonics created by the SIESTA group. [Forpy](https://github.com/ylikx/forpy) is used in the NeuralXC compatible SIESTA implementation.

NeuralXC
==============================
[//]: # (Badges)
[![Build Status](https://travis-ci.org/semodi/neuralxc.svg?branch=master)](https://travis-ci.org/semodi/neuralxc)
[![codecov](https://codecov.io/gh/semodi/neuralxc/branch/master/graph/badge.svg)](https://codecov.io/gh/semodi/neuralxc/branch/master)
[![DOI](https://zenodo.org/badge/175675755.svg)](https://zenodo.org/badge/latestdoi/175675755)

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
   If you are using one of SIESTA's arch.make files simply add the above line to `LIBS=`.

- SIESTA can now be compiled as usual (MPI is still supported). Please refer to their manual regarding details about the installation.

### How-to

Examples on how to train and deploy a machine learned functional can be found in [examples/example_scripts/](examples/example_scripts).

#### Model training

To train/fit a functional a set of structures and their associated reference energies is required. These structures need to be provided in an [ASE](https://wiki.fysik.dtu.dk/ase/) formatted `.xyz` or `.traj` file (in this example `training_structures.xyz`). Assuming that Siesta along with its NeuralXC extension was installed, iterative training can be performed by running

`neuralxc iterative training_structures.xyz basis.json hyperparameters.json`

- `basis.json` contains information regarding the basis set as well as the 'driver' program (SIESTA), examples can be found in [examples/inputs/ml_basis/](examples/inputs/ml_basis).   

- `hyperparameters.json` contains the machine learning hyperparameters, examples can be found in [examples/inputs/hyper](examples/inputs/hyper).

- For more options please refer to `neuralxc iterative --help`


#### Model deployment

To deploy a trained model in SIESTA simply add the line `neuralxc $PATH_TO_NXC_MODEL` (replace with actual path!) to your `.fdf` input file. 

SIESTA can be used with MPI (if compiled accordingly) but the NeuralXC part will be computed in serial. To enable multi-threading in NeuralXC, the option `neuralxc.threads $N_THREADS` can be added to the `.fdf` input file.

 

### Reproducibility 

Data (raw data, input files, trained models) needed to reproduce the results presented in \[2\] can be found in [examples/](examples).

Data that has been obtained from other sources and which has been published in the past is not provided. However, where possible, we have included scripts to download data from the respective repositories.


### Troubleshooting

If the model fails during deployment (i.e. SIESTA crashes when using NeuralXC) try the following steps:

- Run `siesta < $INPUT_FILE_FDF`, i.e. run your calculation while printing the ouput to your screen in real time. When piping the output to a file, siesta does not print python exceptions which makes debugging harder.

- If the model crashes right after `Initializing NeuralXC from Fortran`, make sure you have activated the anaconda environment that contains your NeuralXC installation (if applicable).
If error persists, try to recompile SIESTA while being in the same anaconda environment in which you installed NeuralXC. Compiling SIESTA with different libraries/compilers than are accessible to NeuralXC can cause compatibilty issues.

- If the model crashes after `NeuralXC: Load pipeline from `, double check to make sure the path to the NeuralXC model you provided is correct.

- If problems persist, please create an Issue on GitHub or contact me directly.


### Reference

If you use this code in your work, please cite it as 

[1]  *Machine Learning Accurate Exchange and Correlation Functionals of the Electronic Density.* Sebastian Dick and Marivi Fernandez-Serra, Nature Communications 11, 3509 (2020)

### Copyright

Copyright (c) 2019, Sebastian Dick


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0. 
This project uses an implementation of the gradient of spherical harmonics created by the SIESTA group. [Forpy](https://github.com/ylikx/forpy) is used in the NeuralXC compatible SIESTA implementation.

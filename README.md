NeuralXC
==============================
[//]: # (Badges)
[![Build Status](https://travis-ci.org/semodi/neuralxc.svg?branch=master)](https://travis-ci.org/semodi/neuralxc)
[![codecov](https://codecov.io/gh/semodi/neuralxc/branch/master/graph/badge.svg)](https://codecov.io/gh/semodi/neuralxc/branch/master)


<img src="https://github.com/semodi/neuralxc/blob/master/NeuralXC.png" width="700" height="450" />

Implementation of a machine learned density functional


### Installation

Installation of NeuralXC requires anaconda.
To install NeuralXC, navigate into the root directory of the repository and run 
```
sh install.sh 
```
The installation can be tested by running
```
pytest -v
``` 
in the same directory. 

### Reference

If you use this code in your work, please cite it as 

*Dick, Sebastian, and Marivi Fernandez-Serra. "Learning from the density to correct total energy and forces in first principle simulations." The Journal of Chemical Physics 151.14 (2019): 144102.*

and


*Dick, Sebastian, and Marivi Fernandez-Serra. "Machine Learning a Highly Accurate Exchange and Correlation Functional of the Electronic Density". ChemRxiv 9947312 (preprint), doi:10.26434/chemrxiv.9947312.v1*

### Copyright

Copyright (c) 2019, Sebastian Dick


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.

#!/bin/bash

if [ -z "$python_version" ]
then
    echo "Using python 3.6 by default"
    export python_version=3.6
else
    echo "Using python "$python_version". But recommended to use python 3.6."
fi

if [ -z "$1" ]
then
    echo "Installing NeuralXC in current env"
else
    export envname=$1
    conda create -y --name $envname python=$python_version
    source activate $envname
fi

if ! [ -x "$(command -v pip)" ]; then
	conda install -y pip
fi
yes | pip install --upgrade pip

conda install -y -c pyscf -c defaults -c conda-forge  \
   ase \
   dask \
   h5py=2.9.0 \
   hdf5=1.10.4 \
   ipyparallel \
   ipython \
   jupyter \
   keras \
   matplotlib \
   numba \
   numpy \
   pandas \
   periodictable \
   scikit-learn=0.20 \
   scipy \
   seaborn \
   statsmodels \
   sympy \
   tensorflow \
   pytest \
   pytest-cov \
   pyscf 

yes |  pip install codecov
yes |  pip install tabulate
export NPY_DISTUTILS_APPEND_FLAGS=1
pip install -e .


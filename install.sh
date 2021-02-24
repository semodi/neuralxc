#!/bin/bash

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

conda install -y  -c conda-forge -c defaults  \
   ase=3.17 \
   dask \
   h5py=2.9.0 \
   hdf5=1.10.4 \
   ipyparallel \
   ipython \
   jupyter \
   matplotlib \
   numpy\
   pandas \
   periodictable \
   scikit-learn=0.20.3 \
   scipy \
   pytest \
   pytest-cov \
   dill=0.3.2

conda install -y pytorch=1.4 torchvision torchaudio cpuonly -c pytorch
yes |  pip install opt_einsum 
yes |  pip install codecov
yes |  pip install pyscf
yes |  pip install tabulate
pip install -e .


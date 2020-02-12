# Training a NeuralXC functional

To train a functional adiabatically for ethanol simply run 

`
sh train_ethanol.sh
`
----------------------
The script does the following:

1. Create a work directory and populate with relevant files
1. Download the reference data from sgdml.org and select a small subsets for training (10 samples) and testing (10 samples)
1. Train a NeuralXC functional for Ethanol (@CCSD(T), cc-pvtz). The relevant command for this is :
        
    `neuralxc adiabatic train.traj basis_sgdml_ethanol.json hyperparameters.json --maxit 8 --tol 0.0005 --scale_targets --scale_exp 3`
    
    This adiabatically trains the functional on the reference data contained in `train.traj` (.traj is a ASE native trajectory file) using the basis and dft parameters specified in `basis_sgdml_benzene.json` and the hyperparameters specified in `hyperparameters.json`. `--maxit` sets the maximum number of iterations.
    

This is only a demonstration and the resulting functional won't be as accurate as reported in the manuscript. For more reliable results, the hyperparameters should be picked from a larger grid (see `/inputs/hyper/hyperparameters.json`), `sgdml.fdf` should be replaced by `/inputs/fdf/sgdml.fdf`, and the full test set should be used for evaluation.






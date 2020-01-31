# Training a NeuralXC functional

To train a functional for benzene simply run 

`
sh train_benzene.sh
`
----------------------
The script does the following:

1. Create a work directory and populate with relevant files
1. Download the reference data from sgdml.org and select a small subsets for training (10 samples) and testing (10 samples)
1. Train a NeuralXC functional for Benzene (@CCSD(T), cc-pvdz). The relevant command for this is :
        
    `neuralxc iterative train.traj basis_sgdml_benzene.json hyperparameters.json --maxit 5 --tol 0.0005`
    
    This iteratively trains the functional on the reference data contained in `train.traj` (.traj is a ASE native trajectory file) using the basis and dft parameters specified in `basis_sgdml_benzene.json` and the hyperparameters specified in `hyperparameters.json`. `--maxit` sets the maximum number of iterations, however the training should converge within the tolerance `--tol` (in eV) before reaching 5 iterations.
    
1. Test the trained model on the test-set
1. Save the trained model in `final_model` along with train/test statistics (both in .csv and .html format)

    The test performance should be similar to this :
    ``{'mae': 0.0046, 'max': 0.0108, 'mean deviation': -0.0, 'rmse': 0.0057}`` 
    Slight variations are possible due to the randomness of the fitting procedure.


This is only a demonstration and the resulting functional won't be as accurate as reported in the manuscript. For more reliable results, the hyperparameters should be picked from a larger grid (see `/inputs/hyper/hyperparameters.json`), `sgdml.fdf` should be replaced by `/inputs/fdf/sgdml.fdf`, and the full test set should be used for evaluation.

The expected run-time of `train_benzene.sh` is about 15 min but will be significantly longer if these changes are implemented. 





# Create a work directory 
mkdir workdir
cd workdir 
cp ../* .

# Download reference data from sgdml.org
wget http://quantum-machine.org/gdml/data/xyz/ethanol_ccsd_t.zip
unzip ethanol_ccsd_t.zip

# Extract subsets from reference data for training and testing
python ../../../scripts/apply_subset.py ethanol_ccsd_t-train.xyz ../../../configs/sGDML/ethanol/10.csv train.traj
python ../../../scripts/apply_subset.py ethanol_ccsd_t-test.xyz test_subset.csv testing.traj

# Copy relevant files to work directory
cp ../basis_sgdml_ethanol.json .
cp ../hyperparameters.json .
cp ../sgdml.fdf .
cp ../../../inputs/psf/*.psf .

# basis_sgdml_benzene.json must contain absolute paths for now -> temporary fix:
python ../../../scripts/fix_paths.py ./basis_sgdml_ethanol.json

# Run iterative training
neuralxc adiabatic train.traj basis_sgdml_ethanol.json hyperparameters.json --maxit 8 --tol 0.0005 --scale_targets --scale_exp 3
cd -
cp -r workdir/final_model . 



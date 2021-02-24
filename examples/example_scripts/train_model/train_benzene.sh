# Create a work directory 
mkdir workdir
cd workdir 
cp ../* .

# Download reference data from sgdml.org
wget http://quantum-machine.org/gdml/data/xyz/benzene_ccsd_t.zip
unzip benzene_ccsd_t.zip

# Extract subsets from reference data for training and testing
python ../../../scripts/apply_subset.py benzene_ccsd_t-train.xyz ../../../configs/sGDML/benzene/10.csv train.traj
python ../../../scripts/apply_subset.py benzene_ccsd_t-test.xyz test_subset.csv testing.traj

# Copy relevant files to work directory
cp ../basis_sgdml_benzene.json .
cp ../hyperparameters.json .

# basis_sgdml_benzene.json must contain absolute paths for now -> temporary fix:
python ../../../scripts/fix_paths.py

# Run iterative training
neuralxc sc train.traj basis_sgdml_benzene.json hyperparameters.json --maxit 5 --tol 0.0005
cd -
cp -r workdir/final_model . 
cp -r workdir/final_model.jit . 



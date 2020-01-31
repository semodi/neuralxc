# Create a work directory 
mkdir workdir
cd workdir 
cp ../* .

# Download reference data from sgdml.org
wget http://quantum-machine.org/gdml/data/xyz/benzene_ccsd_t.zip
unzip benzene_ccsd_t.zip

# Extract subsets from reference data for training and testing
python apply_subset.py benzene_ccsd_t-train.xyz ../../../configs/sGDML/benzene/10.csv train.traj
python apply_subset.py benzene_ccsd_t-test.xyz test_subset.csv testing.traj

# Copy relevant files to work directory
cp ../../../inputs/ml_basis/basis_sgdml_benzene.json .
cp ../hyperparameters.json .
cp ../sgdml.fdf .
cp ../../../inputs/psf/*.psf .

# basis_sgdml_benzene.json must contain absolute paths for now -> temporary fix:
python ../fix_paths.py

# Run iterative training
neuralxc iterative train.traj basis_sgdml_benzene.json hyperparameters.json --maxit 1 --tol 0.0005

cd -


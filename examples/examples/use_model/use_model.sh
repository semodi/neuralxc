# Create a work directory 
mkdir workdir_dimer
cd workdir_dimer
cp ../mbpol_dimer.fdf .
cp ../../../inputs/psf/* .

# Add "neuralxc path_to_model" to fdf file to tell SIESTA where to find NeuralXC functional  
# Can use both relative and absolute paths
siesta < mbpol_dimer.fdf | tee 2h2o.out
cd -


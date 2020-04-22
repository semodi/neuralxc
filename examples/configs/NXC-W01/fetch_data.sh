#!/bin/bash
mkdir downloaded
cd downloaded

wget https://pubs.acs.org/doi/suppl/10.1021/ct400863t/suppl_file/ct400863t_si_001.zip
unzip ct400863t_si_001.zip
mv supporting-information 2b

for file in ts-ccpol  ts-pimd-run105  ts-pimd-run107  ts-stationary
do
python ../../../scripts/apply_subset.py 2b/${file}.xyz ../2b_testing/${file}.csv  ${file}_training.traj
done

python ../../../scripts/merge_trajs.py ts-*.traj 
mv merged.traj 2b_testing.traj
rm ts*.traj

wget https://pubs.acs.org/doi/suppl/10.1021/ct500079y/suppl_file/ct500079y_si_001.zip
mkdir 3b 
mv ct500079y_si_001.zip 3b
cd 3b 
unzip ct500079y_si_001.zip
cd ..

python ../../../scripts/apply_subset.py 3b/trimers.xyz ../3b_testing.csv  3b_testing.traj
python ../../../scripts/apply_subset.py 3b/trimers.xyz ../3b_training.csv  3b_training.traj

python ../../../scripts/apply_subset.py ../dimers_training.xyz all dimers_training.traj 1.0
python ../../../scripts/apply_subset.py ../monomers_training.xyz all monomers_training.traj 1.0
python ../../../scripts/apply_subset.py ../monomers_testing.xyz all 1b_testing.traj 1.0
python ../../../scripts/merge_trajs.py monomers_training.traj dimers_training.traj 
mv merged.traj monomers_dimers_training.traj
rm monomers_training.traj
rm dimers_training.traj

rm -r 3b 
rm -r 2b 
rm *.zip
cd ..


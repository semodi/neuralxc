#!/bin/bash
if [ "$#" -eq 0 ]
then
	systems=(benzene toluene malonaldehyde ethanol)
else
	systems=$@
fi

for dir in "${systems[@]}"
do
	cd ${dir}
	wget http://quantum-machine.org/gdml/data/xyz/${dir}_ccsd_t.zip
	unzip ${dir}_ccsd_t.zip
	for i in *.csv
	do
		base=$(basename $i .csv)
		python ../../../scripts/apply_subset.py ${dir}_ccsd_t-train.xyz ${i} ${base}_train.traj
		python ../../../scripts/apply_subset.py ${dir}_ccsd_t-test.xyz all testing.traj
	done
	cd ..
done


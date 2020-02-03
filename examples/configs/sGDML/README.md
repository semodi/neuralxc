# sGDML + Water

Benzene, Toluene, Ethanol and Malonaldehyde data can be downloaded [here](http://quantum-machine.org/gdml/)
Data for Water is contained in the MOB-ML dataset (see `configs/MOB-ML/README.md` for details)

The directories contain `*.csv` indicating which subsets of the original data were used in Fig. 2 of the publication.

Invoking 

`bash fetch_data.sh system_names` (e.g. `bash fetch_data.sh benzene toluene`)  

will download the specified systems from http://quantum-machine.org/gdml/ and apply the subsets. If no system names are specified, all datasets (except water as it is not contained in the sGDML data) are downloaded.

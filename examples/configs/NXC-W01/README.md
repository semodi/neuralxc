# Two-body energies

Two-body data can be downloaded [here](https://pubs.acs.org/doi/10.1021/ct400863t)

Ref: [1] J. Chem. Theory Comput. 2013, 9, 12, 5395-5403

The supporting information of [1] contains four relevant files:

- ts-ccpol.xyz
- ts-pimd-run105.xyz
- ts-pimd-run107.xyz
- ts-stationary.xyz

These files contain dimer structures together with their 2-body energies (in kcal/mol). 
`*.csv` files in the `2b_testing` directory can be used to extract the test set from these files. 
`dimers_training.xyz` also contains structures taken from [1], however, as we used total energies in the fitting of the model rather than 2-body energies we have opted to provide the data ourselves. Total dimer energies were used by combining the 2-body energies from [1] with 1-body energies calculated with the Partridge-Schwenke potential energy surface (J. Chem. Phys. 106, 4618 (1997); https://doi.org/10.1063/1.473987).

# Three-body energies

Three-body data can be downloaded [here](https://pubs.acs.org/doi/10.1021/ct500079y)
Ref: [2] J. Chem. Theory Comput. 2014, 10, 4, 1599-1607

`3b_training.csv` and `3b_testing.csv` can be used extract train and test sets from `trimers.xyz` (which is contained in the supporting information of [2]).
`trimers.xyz` contains trimer structures together with their 3-body energies (in kcal/mol). 

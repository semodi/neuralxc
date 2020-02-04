# Use a trained NeuralXC functional

`sh use_model.sh` will use the water functional (trained on MB-pol dataset) on a water dimer 

The only difference to a standard SIESTA calculation is an additional line in the 'mbpol_dimer.fdf' input file (last line):

`neuralxc ../../../models/MB-pol/model`

This line instructs SIESTA to use the NeuralXC model stored in `../../../models/MB-pol/model` (both absolute and relative paths can be used). Note that as the model was built as an additive correction to PBE, we still have to keep GGA/PBE as the xc-functional option.

The calculation is done in serial model and should take about 1 min. 
The following final energies should be found in `workdir_dimer/2h2o.out`:

```
siesta: Final energy (eV):
siesta:  Band Struct. =    -222.933172
siesta:       Kinetic =     702.847181
siesta:       Hartree =    1129.631537
siesta:       Eldau   =       0.000000
siesta:       Eso     =       0.000000
siesta:    Ext. field =       0.000000
siesta:       Enegf   =       0.000000
siesta:   Exch.-corr. =    -236.953949
siesta:  Ion-electron =   -2875.395071
siesta:       Ion-ion =     339.069754
siesta:       Ekinion =       0.000000
siesta:         Total =    -940.800547
siesta:         Fermi =      -4.678726
```

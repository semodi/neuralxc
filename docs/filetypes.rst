
Filetypes
=============

NeuralXC uses three different file types.

**System geometries** can either be provided as a ``.xyz`` or an atomic simulation environment (ASE) native ``.traj`` file.
Both file types should be readable by ASE and if energies are needed, which is almost always the case, the
files need to be formatted so that the ASE method ``get_potential_energy()`` returns the appropriate value.

**Configuration files** need to be provided in a ``.json`` format. Two different kinds of configuration files are used within NeuralXC.
**config.json** defines the projection basis set as well as instructions to the engine (the electronic structure code).
**hyper.json** contains the hyperparameters used for the machine learning model.
Both configuration file types are discussed in detail further below.

**Processed data** is generally stored in hierarchical data format ``.hdf5`` files.
These files are subdivided by groups which can (but do not have to) indicate the data set name (e.g. "ethanol")
and the method used to generate the data (e.g. "PBE" or "CCSD") separated by a "/".

The content of a typical file will look like this::

      water/PBE/energy
      water/PBE/density/05e5598680faeecf6f5d8ebe4283e76d
      water/PBE/forces
      water/CCSD/energy
      ethanol/PBE/energy
      ethanol/PBE/density/05e5598680faeecf6f5d8ebe4283e76d
      ethanol/CCSD/energy

We will refer to this example file when discussing :ref:`CLI` commands.
The hash-codes shown act as unique identifiers for the basis set chosen to conduct the density
projection and are automatically created from the content of the ``config.json`` input file.

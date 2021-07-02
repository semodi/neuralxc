.. _CLI:

CLI
============

The recommended way for users to interact with NeuralXC is through the command line interface (CLI).
While the self-consistent training command ``neuralxc sc`` introduced above should cover 95% of all use cases,
sometimes a more fine-grained control flow is warranted.
This can be achieved by utilizing the following commands which can be grouped into three categories:  Data, Model and Other.

Data
---------

Commands in this category deal with managing data, i.e. input features along with target energies.  All data-related commands are prefaced with::

    neuralxc data

so that, e.g. in order to add data to an ``.hdf5`` file, the command::

    neuralxc data add <args>

needs to be executed.

We provide a complete list of these commands, with required arguments shown within the command prompt and optional arguments listed underneath.

**add**
::
  add <hdf5> <system> <method> <add>

Store data to file ``<hdf5>`` under the group ``<system>/<method>/``. The quantity to add is specified as ``<add>`` and can be either ``energy``,
``forces`` or  ``density``. If adding energies or forces  ``--traj <str>`` needs to be set to point to an  ``.xyz`` or ``.traj``
file containing the required quantity. If adding densities, ``--density <str>`` needs to be set to the path were density projections are stored.

   ``--zero <float>``  Shift energies  by this value. If not set, shifts energies so that minimum of dataset is zero.

    **Example:** ``neuralxc data add data.hdf5 water PBE energy --traj water_pbe.traj``

**delete**
::
  delete <hdf5> <group>

Delete data from file ``<hdf5>`` within group ``<group>``. Cannot be reversed.

    **Example:** ``neuralxc data delete water/PBE``

**split**
::
  split <hdf5> <group> <label>

Split slice off data from file  ``<hdf5>`` within group ``<group>``.
Slicing can be provided in numpy notation by setting  ``--slice <str>``. ``--comp <str>`` stores the complementary slice as its own dataset

  **Example:** ``neuralxc data split data.hdf5 water/PBE training --slice :15 --comp testing``

  This splits of the first 15 datapoints from water/pbe stored in data.hdf5, stores it as training and stores the remaining datapoints as testing.

**sample**
::
  sample <config> <size> --hdf5 <hdf5> --dest <dest>

Sample ``<size>`` data points for the basis set defined in ``<config>`` from ``<hdf5>``,
saving it to ``<dest>`` using k-means clustering in feature space.

    **Example:** ``neuralxc data sample config.json 50 --hdf5 data.hdf5/water/PBE --dest sample_50.npy``

Model
---------

Commands in this category deal with the machine learning model, they are prefaced with
::
  neuralxc


**fit**
::
  fit <config> <hyper> --hdf5 <path> <baseline> <reference>

Use features generated with basis defined in  ``<config>`` and hyperparameters defined in  ``<hyper>`` to fit a neuralxc model that corrects
``<baseline>`` data in hdf5 file found at  ``<path>`` using targets given by  ``<reference>``.

  ``--model <str>`` Continue training model found at this location
  ``--hyperopt`` If set, conduct hyperparameter optimization.

  **Example:**  ``neuralxc fit config.json hyper.json --hdf5 data.hdf5 water/PBE water/CCSD``

**eval**
::
  eval --hdf5 <path> <baseline> <reference>

Evaluate accuracy of  ``<baseline>`` with respect to ``<reference>``

    ``--model <str>`` If set, correct baseline with this model before evaluation.
    ``--plot`` Create error histogram and correlation plot.
    ``--sample <str>`` Only evaluate on this sample (.npy file containing integer indices)
    ``--keep_mean`` If set, don't subtract parallelity errors.

    **Example:**  ``neuralxc eval --hdf5 data.hdf5 water/PBE water/CCSD --model best_model``

**predict**
::
  predict --model <model> --hdf5 <hdf5>

Predict energy corrections to data in ``<hdf5`` using ``<model>``.

    ``--dest <str>`` Store to this location (default: prediction.npy)

    **Example:**  ``neuralxc predict --model best_model --hdf5 data.hdf5/water/PBE``

**serialize**
::
  serialize <in_path> <jit_path>

Serialize model found at ``<in_path>`` and store to ``<jit_path>`` to be used with libnxc.

    ``--as_radial`` serializes model to be used with radial grids.


Other
--------
Commands in this category deal with running and processing SCF calculations, they are prefaced with
::
    neuralxc

**engine**
::
     engine <config> <xyz>

Run engine (electronic structure code) specified in ``<config>`` for all molecules contained in ``<xyz>``.
Stores results (energies) of calculations in ``results.traj``

    ``--workdir <str>``  Specify work-directory. Default is to use .tmp/ and delete after calculation has finished

**default**
::
  default <kind>

Generates a default input file either containing basis set information (``<kind> = pre``) or hyperparameters (``<kind> = hyper``)

**preprocess**
::
  pre <config> --xyz <xyz> --dest <dest> --srcdir <srcdir>

Preprocesses (projects) electron densities found at ``<srcdir>`` for systems found in the ``<xyz>`` .xyz or .traj file and stores features in `` <dest>`` (a hdf5 file path with group name).

  **Example:**  ``neuralxc pre config.json --xyz water_pbe.traj --dest data.hdf5 water/PBE --srcdir workdir/``

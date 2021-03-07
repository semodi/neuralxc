.. _Input:

Input Files
===================

Most NeuralXC CLI commands require two types of input files:

- preprocessor file : Contains information regarding the density projection basis, as well
as the projector and symmetrizer type. Instructions on how to run SCF calculations
such as the type of driver code as well as the basis sets etc. should also be contained.

- hyperparameters file : Contains the hyperparameters used the machine learning
  model (anything following symmetrization). Details are provided below.


Preproccesor
-------------

Let's start with an example:

pyscf with **analytical** gaussian projectors::

    {
      "preprocessor":
      {
           "basis": "cc-pVTZ-jkfit",
           "extension": "chkpt",
           "application": "pyscf",
           "projector_type": "pyscf",
           "symmetrizer_type": "trace"

      },
      "n_workers" : 1,
      "engine_kwargs": {"xc": "PBE",
                        "basis" : "cc-pVTZ"}

    }

"n_workers" determines over how many processes SCF
calculations and subsequent density projections are distributed.
Apart from this, there are two groups:

- "preprocessor" contains the necessary information to project the density and
  symmetrize it.
- "engine_kwargs" determines the behavior of the electronic structure code
  specified in preprocessor[application].

In our example we use PySCF and project the density analytically onto
Gaussian type orbitals with the "pyscf" projector using the "cc-pVTZ-jkfit"
basis . For a selection of different projectors, see :ref:`Projector`.


SIESTA with **numerical** polynomial projectors::

  {
      "preprocessor": {
          "C": {
              "n": 4,
              "l": 5,
              "r_o": 2
          },
          "H": {
              "n": 4,
              "l": 5,
              "r_o": 2
          },
          "extension": "RHOXC",
          "applications": "siesta",
          "projector_type": "ortho",
          "symmetrizer_type": "trace"

      },
      "src_path": "workdir",
      "n_workers": 1,
      "engine_kwargs": {"pseudoloc" : ".",
                        "fdf_path" : null,
                        "xc": "PBE",
                        "basis" : "DZP",
                        "fdf_arguments": {"MaxSCFIterations": 50}
  }

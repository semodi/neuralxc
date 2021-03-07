.. _Projector:

Projector
=============

Base classes and inheritance
-----------------------------

All real space projectors are either derived from EuclideanProjector or RadialProjector. The former
implements density projections on a euclidean grid with periodic boundary conditions. The latter can
be used for projections on flexible grids for which grid coordinates and integration weights are explicitly
provided (as e.g. in PySCF). Both of these projector types inherit from the BaseProjector class.

.. autoclass:: neuralxc.projector.projector.BaseProjector
   :members:  get_basis_rep

.. autoclass:: neuralxc.projector.projector.EuclideanProjector
   :members: __init__

.. autoclass:: neuralxc.projector.projector.RadialProjector
   :members: __init__

Radial basis
---------------

Starting from from these definitions, NeuralXC implements two projectors that differ in their radial basis
functionals. OrthoProjector implements an orthonormal polynomial basis whereas GaussianProjector uses
Gaussian type orbitals similar to those used in quantum chemistry codes. Both projectors come in a euclidean
and radial version.


.. autoclass:: neuralxc.projector.polynomial.OrthoProjector
.. autoclass:: neuralxc.projector.polynomial.OrthoRadialProjector

.. autoclass:: neuralxc.projector.gaussian.GaussianProjector
.. autoclass:: neuralxc.projector.gaussian.GaussianRadialProjector


PySCF projector
-----------------

If GTO orbitals in both projection and DFT calculation, projection integrals can be computed
analytically. For this purpoes we have implemented a projector that works with PySCF. Future version of NeuralXC will
implement a more general density matrix projector class that works with other gto codes as well.

.. autoclass:: neuralxc.projector.pyscf.PySCFProjector
  :members: __init__, get_basis_rep

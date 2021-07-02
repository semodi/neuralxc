.. _Projector:

Projector
=============

A crucial step in NeuralXC is the creation of machine learning features :math:`c_\alpha` by projecting the density :math:`n(\mathbf r)` onto a set of basis functions :math:`\psi_{\alpha}(\mathbf r)`

.. math::
    c_\alpha = \int_\mathbf r \psi_\alpha(\mathbf r) n(\mathbf r).

Here, :math:`\alpha` summarizes the quantum numbers :math:`n,l,m`.
In practice, this integration is either performed on a grid, or, if both density matrix and projection basis can be expressed in terms of Gaussian type orbitals (GTO), analytically.

For grid integrations, NeuralXC distinguishes between three dimensional euclidean grids

.. math::
    c_\alpha = V_\text{grid} \cdot \sum_x\sum_y\sum_z \psi_{\alpha, xyz} \cdot n_{xyz}

and radial grids

.. math::
    c_\alpha = \sum_i \psi_{\alpha, i} \cdot n_{i}\cdot w_i

with integration weights :math:`w_i`. For euclidean grids, NeuralXC automatically assumes periodic boundary conditions. Radial grids are only supported for finite systems.

If the density can be expressed in terms of GTOs :math:`\phi(\mathbf r)` with density matrix :math:`\rho_{\mu\nu}`

.. math::
    n(\vec r) = \sum_{\mu\nu}\rho_{\mu\nu }\phi_\mu(\mathbf r) \phi_\nu(\mathbf r)

then the projection can be computed analytically as

.. math::
    c_\alpha = \sum_{\mu\nu}\rho_{\mu\nu } \int_\mathbf r \psi_\alpha(\mathbf r) \phi_\mu(\mathbf r) \phi_\nu(\mathbf r).

NeuralXC takes advantage of this to avoid grid computations when working with PySCF \cite{pyscf}.

Beyond the type of integration methods, Projectors are defined by their radial basis functions (the angular part is always given by spherical harmonics).

These can either be polynomials

.. math::
  \zeta_n(r) = \left\{
                  \begin{array}{ll}
                  \frac{1}{N}r^2(r_o -r)^{n+2}
                  %\exp(-\gamma (\frac{r}{r_o})^{1/4})
  \text{ for } r < r_o \\
   &\text{ else }
  		\end{array}
  			      \right.


or GTOs.

For polynomial basis functions the user needs to specify the number of
radial functions :math:`n`, the number of spherical shells :math:`l` (this is the maximum angular momentum :math:`l_\text {max}` plus one)
and an outer cutoff :math:`r_o` in the configuration file. Basis sets can either be specified on a per-element basis or a single,
species independent, basis set can be used across elements.

For GTO basis functions, the user can either provide a basis-set name (same nomenclature as used by PySCF) such as "6-311G*" or a path pointing to a file containing a NWChem \cite{nwchem} formatted basis definition (see Basis Set Exchange \cite{bse}). If the projection is done on a grid (as opposed to analytically) the Gaussians should be localized to have finite support

.. math::
  r_o &= \sigma \cdot \alpha^{1/2}  \cdot ( 1 + \frac{l}{5}) \\
  \tilde r &= \frac{r}{\gamma}\\
  f_c(r) &= 1 - \left[0.5 \cdot \left\{1 - \cos(\pi \frac{ r}{r_o})\right\}\right]^8\\
  \zeta(r) &= \left\{
                  \begin{array}{ll}
                  \tilde r^l\exp(-\kappa \tilde r^2) \cdot f_c(\tilde r)
                  %\exp(-\gamma (\frac{r}{r_o})^{1/4})
  &\text{ for } r < r_o \\
  0 &\text{ else }
  		\end{array} \right.

We have introduced two parameters that can be tuned by the user to control the localization of the basis functions.
:math:`\sigma` (default: 2.0) controls the effective cutoff radius as a function of the Gaussian exponent :math:`\kappa`
and the angular momentum :math:`l`. :math:`\gamma` (default: 1.0) allows the user to re-scale the Gaussian functions.

The type of radial basis to be used can be specified in the configuration file using the keyword

``projector``
    - ``ortho``  Polynomial basis
    - ``gaussian``  GTO basis


In addition, the grid (or lack thereof) has to be specified through the keyword 

``grid``
   - ``euclidean``  Euclidean grid
   - ``radial``     Radial grid
   - ``analytical`` Perform integrals analytically

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


Symmetrizer
===============

All Symmetrizer classes are derived from BaseSymmetrizer

.. autoclass:: neuralxc.symmetrizer.symmetrizer.BaseSymmetrizer
   :members: __init__, get_symmetrized


Customized Symmetrizers can be created by inheriting from this base class and
implementing the method `_symmetrize_function`. As of now two symmetrizers
are implemented by default:


.. autoclass:: neuralxc.symmetrizer.symmetrizer.TraceSymmetrizer
   :members: _symmetrize_function

.. autoclass:: neuralxc.symmetrizer.symmetrizer.MixedTraceSymmetrizer
   :members: _symmetrize_function

If

.. math::
  C_{nlm}
is the density projection with principal quantum number n, angular momentum l,
and angular momentum projection m then trace symmetrizers create a rotationally
invariant feature by taking the trace of the outer product over m of C with itself:

.. math::
  D_{nl} = \text{Tr}_{mm'}[C \otimes_{m} C]

`MixedTraceSymmetrizer` generalizes this approach by mixing radial channels obtaining

.. math::
  D_{nn'l}

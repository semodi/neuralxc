
Symmetrizer
===============

In order for the energy (the model output) to be invariant with respect to global rotations NeuralXC symmetrizes the descriptors :math:`c_{nlm}``.
Two symmetrizers are currently supported by NeuralXC and can be set with the keyword

``symmetrizer_type``
    - ``trace``  :math:`d_{nl} = \sum_m c_{nlm}^2`
    - ``mixed_trace`` :math:`d_{nn'l} = \sum_m c_{nlm}c_{n'lm}``

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

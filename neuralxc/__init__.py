"""
neuralxc
Implementation of a machine learned density functional
"""

# Add imports here
import warnings
warnings.filterwarnings("ignore")
PYSCF_FOUND = True
try:
    import torch
except ModuleNotFoundError:
    TORCH_FOUND = False
try:
    import pyscf
except ModuleNotFoundError:
    PYSCF_FOUND = False
from . import config
from .neuralxc import NeuralXC, PySCFNXC
from .neuralxc import NeuralXC as NeuralXCJIT

from . import pyscf
from . import projector
from . import utils
from . import constants
from . import symmetrizer
from . import ml
from . import base
from . import datastructures
from . import drivers

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

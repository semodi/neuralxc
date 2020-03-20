"""
neuralxc
Implementation of a machine learned density functional
"""

# Add imports here
import warnings

from . import (base, constants, datastructures, drivers, ml, projector, pyscf,
               symmetrizer, utils)
# Handle versioneer
from ._version import get_versions
from .neuralxc import NeuralXC, SiestaNXC, get_nxc_adapter, get_V, verify_type

warnings.filterwarnings("ignore")
PYSCF_FOUND=True
try:
    import pyscf
except ModuleNotFoundError:
    PYSCF_FOUND=False
# from . import formatter
#from .projector import *
#from .mlpipeline import *
#from .symmetrizer import *

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

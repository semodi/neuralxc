try:
    from .pyscf import RKS
    from ..projector.pyscf import BasisPadder
except ModuleNotFoundError:
    print('PySCF not found')
    pass

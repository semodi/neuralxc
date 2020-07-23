from abc import ABC, abstractmethod
from ..base import ABCRegistry
try:
    from ..pyscf.pyscf import compute_KS
    from pyscf.scf.chkfile import load_scf
except ModuleNotFoundError:
    compute_KS = None
from ase.calculators.singlepoint import SinglePointCalculator
from .siesta import CustomSiesta
import os
from ase.units import Hartree


class EngineRegistry(ABCRegistry):
    REGISTRY = {}


class BaseEngine(metaclass=EngineRegistry):

    _registry_name = 'base'

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute(self):
        pass


def Engine(app, **kwargs):

    registry = BaseEngine.get_registry()
    if not app in registry:
        raise Exception('Egnine: {} not registered'.format(app))

    return registry[app](**kwargs)


class PySCFEngine(BaseEngine):

    _registry_name = 'pyscf'

    def __init__(self, **kwargs):
        self.xc = kwargs.get('xc', 'PBE')
        self.basis = kwargs.get('basis', 'ccpvdz')
        self.nxc = kwargs.get('nxc', '')
        self.skip_calculated = kwargs.pop('skip_calculated', True)

    def compute(self, atoms):
        if 'pyscf.chkpt' in os.listdir('.') and self.skip_calculated:
            mol, results = load_scf('pyscf.chkpt')
            e = results['e_tot']
        else:
            mf, mol = compute_KS(atoms, basis=self.basis, xc=self.xc, nxc=self.nxc)
            e = mf.energy_tot()

        atoms.calc = SinglePointCalculator(atoms)
        atoms.calc.results = {'energy': e * Hartree}
        return atoms

class PySCFEngine(PySCFEngine):

    _registry_name = 'pyscf_rad'


class SiestaEngine(BaseEngine):

    _registry_name = 'siesta'

    def __init__(self, **kwargs):
        fdf_path = kwargs.pop('fdf_path', None)

        # Defaults
        kwargs['label'] = kwargs.get('label', 'siesta')
        kwargs['xc'] = kwargs.get('xc', 'PBE')
        kwargs['basis_set'] = kwargs.pop('basis', 'DZP')
        kwargs['fdf_arguments'] = kwargs.get('fdf_arguments', {'MaxSCFIterations': 200})
        kwargs['pseudo_qualifier'] = kwargs.get('pseudo_qualifier', '')
        exec_prepend = kwargs.pop('exec_prepend', '')

        # Environment variables for ase
        os.environ['SIESTA_PP_PATH'] = kwargs.pop('pseudoloc', '.')
        if not 'SIESTA_COMMAND' in os.environ:
            os.environ['SIESTA_COMMAND'] = exec_prepend + ' siesta < ./%s > ./%s'

        self.calc = CustomSiesta(fdf_path, **kwargs)

    def compute(self, atoms):
        atoms.calc = self.calc
        atoms.get_potential_energy()
        return atoms

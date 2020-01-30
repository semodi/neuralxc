from ase.io import read, write
import numpy as np
from ase.units import kcal, mol
from ase.calculators.singlepoint import SinglePointCalculator 
import sys
kcalpmol = kcal/mol

def get_structures_energies(path, unit = 1):
    """
    Loads an xyz file that contains energies in the comment line 
    and returns an ase Atoms list
    """
    content = np.genfromtxt(path, delimiter ='###')
    energies = content[1::int(content[0]+2)] * unit

    atoms = read(path,':')

    for a, e in zip(atoms, energies):
        a.calc = SinglePointCalculator(a)
        a.calc.results['energy'] = e

    return atoms


if __name__ == '__main__':
    
    atoms = get_structures_energies(sys.argv[1], unit = kcalpmol)
    subset = np.genfromtxt(sys.argv[2]).astype(int)
    atoms_subset = [atoms[s] for s in subset]
    write(sys.argv[3], atoms_subset)






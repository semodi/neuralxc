from ase.calculators.siesta.siesta import Siesta
from ase.calculators.calculator import Calculator, all_changes
import shutil
import os
from ase.calculators.siesta.base_siesta import *


class CustomSiesta(Siesta):
    def __init__(self, fdf_path=None, **kwargs):
        self.fdf_path = fdf_path
        self.nxc = kwargs.pop('nxc', '')
        self.skip_calculated = kwargs.pop('skip_calculated', True)
        if not self.skip_calculated:
            print('Siesta Caculator is not re-uisng results')
        kwargs.pop('mbe','')
        if 'label' in kwargs:
            kwargs['label'] = kwargs['label'].lower()
        super().__init__(**kwargs)
        self.parameters.update({'symlink_pseudos': None})

    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties, system_changes)

        filename = os.path.join(self.directory, self.label + '.fdf')
        # add custom fdf
        if self.fdf_path:
            with open(self.fdf_path, 'r') as custom_fdf:
                all_custom_keys = [list(entry.keys())[0]\
                 for _, entry in next_fdf_entry(custom_fdf)]

            filename_tmp = os.path.join(self.directory, self.label + '.tmp')
            with open(filename_tmp, 'w') as tmp_file:
                with open(self.fdf_path, 'r') as custom_fdf:
                    tmp_file.write(custom_fdf.read())

                with open(filename, 'r') as ase_fdf:
                    for is_block, entry in next_fdf_entry(ase_fdf):
                        if not list(entry.keys())[0] in all_custom_keys:
                            if 'pao' in list(entry.keys())[0] \
                            and any(['pao' in key for key in all_custom_keys]):
                                continue
                            if is_block:
                                tmp_file.write('%block ')
                                tmp_file.write(list(entry.keys())[0])
                                tmp_file.write('\n')
                                tmp_file.write(list(entry.values())[0])
                                tmp_file.write('%endblock ')
                                tmp_file.write(list(entry.keys())[0])
                                tmp_file.write('\n')
                            else:
                                tmp_file.write(' '.join(list(entry.items())[0]))
                                tmp_file.write('\n')

            with open(filename_tmp, 'r') as tmp_file:
                with open(filename, 'w') as ase_fdf:
                    ase_fdf.write(tmp_file.read())
        if self.nxc:
            with open(filename, 'a') as ase_fdf:
                ase_fdf.write('NeuralXC {} \n'.format(self.nxc))

    def _write_species(self, f, atoms):
        """Write input related the different species.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        species, species_numbers = self.species(atoms)

        if not self['pseudo_path'] is None:
            pseudo_path = self['pseudo_path']
        elif 'SIESTA_PP_PATH' in os.environ:
            pseudo_path = os.environ['SIESTA_PP_PATH']
        else:
            mess = "Please set the environment variable 'SIESTA_PP_PATH'"
            raise Exception(mess)

        f.write(format_fdf('NumberOfSpecies', len(species)))
        f.write(format_fdf('NumberOfAtoms', len(atoms)))

        pao_basis = []
        chemical_labels = []
        basis_sizes = []
        synth_blocks = []
        for species_number, spec in enumerate(species):
            species_number += 1
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                label = symbol
                pseudopotential = label + '.psf'
            else:
                pseudopotential = spec['pseudopotential']
                label = os.path.basename(pseudopotential)
                label = '.'.join(label.split('.')[:-1])

            if not os.path.isabs(pseudopotential):
                pseudopotential = join(pseudo_path, pseudopotential)

            if not os.path.exists(pseudopotential):
                mess = "Pseudopotential '%s' not found" % pseudopotential
                raise RuntimeError(mess)

            name = os.path.basename(pseudopotential)
            name = name.split('.')
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number

            name = '.'.join(name)
            pseudo_targetpath = self.getpath(name)

            if join(os.getcwd(), name) != pseudopotential:
                if islink(pseudo_targetpath) or isfile(pseudo_targetpath):
                    os.remove(pseudo_targetpath)
                symlink_pseudos = self['symlink_pseudos']

                if symlink_pseudos is None:
                    symlink_pseudos = not os.name == 'nt'

                if symlink_pseudos:
                    os.symlink(pseudopotential, pseudo_targetpath)
                else:
                    shutil.copy(pseudopotential, pseudo_targetpath)

            if not spec['excess_charge'] is None:
                atomic_number += 200
                n_atoms = sum(np.array(species_numbers) == species_number)

                paec = float(spec['excess_charge']) / n_atoms
                vc = get_valence_charge(pseudopotential)
                fraction = float(vc + paec) / vc
                pseudo_head = name[:-4]
                fractional_command = os.environ['SIESTA_UTIL_FRACTIONAL']
                cmd = '%s %s %.7f' % (fractional_command, pseudo_head, fraction)
                os.system(cmd)

                pseudo_head += '-Fraction-%.5f' % fraction
                synth_pseudo = pseudo_head + '.psf'
                synth_block_filename = pseudo_head + '.synth'
                os.remove(name)
                shutil.copyfile(synth_pseudo, name)
                synth_block = read_vca_synth_block(synth_block_filename, species_number=species_number)
                synth_blocks.append(synth_block)

            if len(synth_blocks) > 0:
                f.write(format_fdf('SyntheticAtoms', list(synth_blocks)))

            label = '.'.join(np.array(name.split('.'))[:-1])
            string = '    %d %d %s' % (species_number, atomic_number, label)
            chemical_labels.append(string)
            if isinstance(spec['basis_set'], PAOBasisBlock):
                pao_basis.append(spec['basis_set'].script(label))
            else:
                basis_sizes.append(("    " + label, spec['basis_set']))
        f.write((format_fdf('ChemicalSpecieslabel', chemical_labels)))
        f.write('\n')
        f.write((format_fdf('PAO.Basis', pao_basis)))
        f.write((format_fdf('PAO.BasisSizes', basis_sizes)))
        f.write('\n')

    def read_ion(self, atoms):
        """Read the ion.xml file of each specie
        """
        from ase.calculators.siesta.import_ion_xml import get_ion

        species, species_numbers = self.species(atoms)

        self.results['ion'] = {}
        for species_number, spec in enumerate(species):
            species_number += 1

            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                label = symbol
                pseudopotential = self.getpath(label, 'psf')
            else:
                pseudopotential = spec['pseudopotential']
                label = os.path.basename(pseudopotential)
                label = '.'.join(label.split('.')[:-1])

            name = os.path.basename(pseudopotential)
            name = name.split('.')
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number
            name = '.'.join(name)

            label = '.'.join(np.array(name.split('.'))[:-1])

            if label not in self.results['ion']:
                fname = self.getpath(label, 'ion.xml')
                if os.path.isfile(fname):
                    self.results['ion'][label] = get_ion(fname)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):

        if '0_NORMAL_EXIT' in os.listdir('.') and self.skip_calculated:
            Calculator.calculate(self, atoms, properties, system_changes)
            self.write_input(self.atoms, properties, system_changes)
            # if self.command is None:
            # raise CalculatorSetupError(
            # 'Please set ${} environment variable '
            # .format('ASE_' + self.name.upper() + '_COMMAND') +
            # 'or supply the command keyword')
            # command = self.command.replace('PREFIX', self.prefix)
            # errorcode = subprocess.call(command, shell=True, cwd=self.directory)

            # if errorcode:
            #     raise CalculationFailed('{} in {} returned an error: {}'
            #                             .format(self.name, self.directory,
            #                                     errorcode))
            self.read_results()
        else:
            super().calculate(atoms, properties, system_changes)

    def getpath(self, fname=None, ext=None):
        """ Returns the directory/fname string """
        if fname is None:
            fname = self.prefix
        if ext is not None:
            fname = '{}.{}'.format(fname, ext)
        return os.path.join(self.directory, fname)


def next_fdf_entry(file):

    inside_block = False
    block_content = ''
    block_name = ''
    line = file.readline()
    while (line):
        if len(line.strip()) > 0:
            if line.strip()[0] == '%':
                if not inside_block:
                    block_name = ' '.join(line.split()[1:]).lower()
                else:
                    block_out = block_content
                    block_content = ''
                    yield True, {block_name: block_out}

                inside_block = (not inside_block)

            elif not inside_block:
                yield False, {line.split()[0].lower(): ' '.join(line.split()[1:])}
            else:
                block_content += line

        line = file.readline()

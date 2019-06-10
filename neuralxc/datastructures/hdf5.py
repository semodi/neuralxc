from ase.io import read, write
import hashlib
import json

def add_energy(*args, **kwargs):
    return add_data('energy', *args, **kwargs)

def add_forces(*args, **kwargs):
    return add_data('forces', *args, **kwargs)

def add_density(key, *args, **kwargs):
    return add_data(key, *args, **kwargs)

def add_species(file, system, traj_path = ''):
    """
    Add an attribute containing the species string for a given
    system (ex. for water: {'species': 'OHH'})

    Parameters
    ----------

    file: hdf5 file handle
        File to add data to
    system: str
        System label defining first part of group
        in datafile
    """

    order = [system]
    cg = file #Current group
    for idx, o in enumerate(order):
        if not o in cg.keys():
            cg = cg.create_group(o)
        else:
            cg = cg[o]

    if not 'species' in cg.attrs:
        if not traj_path:
            raise Exception('Must provide a trajectory file to define species')
        species = ''.join(read(traj_path, 0).get_chemical_symbols())
        cg.attrs.update({'species' : species})

def add_data(which, file, data, system, method,
              override= False, E0 = None):
    """
    Add data to hdf5 file.

    Parameters
    ----------

    which: str
        Add 'energy', 'forces' or 'density'
    file: hdf5 file handle
        File to add data to
    data: numpy ndarray
        Data to add
    system: str
        System label defining first part of group
        in datafile
    method: str
        Method label defining second part of group
        in datafile
    override: bool
        If dataset already exists in file, override it?
    """

    order = [system, method]
    if not which in ['energy','forces']:
        order.append('density')

    cg = file #Current group
    for idx, o in enumerate(order):
        if not o in cg.keys():
            cg = cg.create_group(o)
        else:
            cg = cg[o]

    if which =='energy':
        if E0 == None:
            cg.attrs.update({'E0': min(data)})
        else:
            cg.attrs.update({'E0': E0})

    print('{} systems found, adding {}'.format( len(data), which))

    def create_dataset():
        cg.create_dataset(which,
                data = data)

    try:
        create_dataset()
    except RuntimeError:
        if override:
            del cg[which]
            create_dataset()
        else:
            print('Already exists. Set override=True')

def basis_to_hash(basis):
    """
    Convert a given basis to a unique identifier

    Parameters
    ---------

    basis: dict
        Contains the basis like so : {'species1': {'n': 1, 'l': 2}...}

    Returns
    --------

    hash: str
        Encoding of the basis set
    """
    return hashlib.md5(json.dumps(basis).encode()).hexdigest()

# def find_unique_root(file, group, path = '/'):
#
#     all_roots = find_root(file, args.group).split('//')
#     if len(all_roots) == 1:
#         raise Exception('No group with name {} found'.format(args.group))
#     if len(all_roots) > 2:
#         raise Exception('Group name must be unique, but {}\
#             matches found for {}'.format(len(all_roots)-1, args.group))
#     root = '/' + all_roots[1]
#     return root
#
# def find_root(file, group, path = '/'):
#     try:
#         matches = ''
#         for key in file[path].keys():
#             if group == key:
#                 return path + '/' + key
#             else:
#                 matches += find_root(file,group, path + '/' + key)
#         return matches
#     except AttributeError:
#         return ''

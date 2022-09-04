import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'basis_sgdml_benzene.json'
print('FILEPATH', path)

basis = json.load(open(path, 'r'))
basis['engine_kwargs']['pseudoloc'] = os.path.abspath(basis['engine_kwargs']['pseudoloc'])
basis['engine_kwargs']['fdf_path'] = os.path.abspath(basis['engine_kwargs']['fdf_path'])

json.dump(basis, open(path, 'w'), indent=4)

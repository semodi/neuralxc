import json
import os
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = 'basis_sgdml_benzene.json'

print('FILEPATH', path)

basis = json.load(open(path, 'r'))
basis['engine_kwargs']['pseudoloc'] = os.path.abspath(basis['engine_kwargs']['pseudoloc'])
basis['engine_kwargs']['fdf_path'] = os.path.abspath(basis['engine_kwargs']['fdf_path'])

json.dump(basis, open(path, 'w'), indent=4)

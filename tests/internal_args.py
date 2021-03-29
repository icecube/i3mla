"""Docstring"""

__author__ = 'John Evans'
__copyright__ = 'Copyright 2020 John Evans'
__credits__ = ['John Evans', 'Jason Fan', 'Michael Larson']
__license__ = 'Apache License 2.0'
__version__ = '0.0.1'
__maintainer__ = 'John Evans'
__email__ = 'john.evans@icecube.wisc.edu'
__status__ = 'Development'

from typing import List

import argparse
import glob
import pickle
import numpy as np
import matplotlib as mpl


def numpy_multifile(glob_strs: List[str]) -> np.ndarray:
    """Docstring"""
    files = [file for glob_str in glob_strs for file in glob.glob(glob_str)]
    arrs = [np.load(file) for file in files]
    return np.concatenate(arrs)


def parse(argv: List[str]) -> dict:
    """Docstring"""
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument(
        '-d', '--data',
        nargs='*',
        default=None,
        help='',
    )
    parser.add_argument(
        '-s', '--sim',
        nargs=1,
        default=None,
        help='',
    )
    parser.add_argument(
        '-g', '--grl',
        nargs='*',
        default=None,
        help='',
    )
    parser.add_argument(
        '-m', '--model',
        nargs=1,
        default=None,
        help='',
    )
    parser.add_argument(
        '-p', '--mpl-backend',
        nargs=1,
        default=None,
        help='',
    )
    parser.add_argument(
        '-o', '--outdir',
        nargs=1,
        default='./',
        help='',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_const',
        const=True,
        default=False,
        help='',
    )

    output = {'plot_inline': False, 'model': None}
    args, _ = parser.parse_known_args(argv)

    if args.model is not None:
        if args.verbose:
            print('Loading model...', end='', flush=True)

        with open(args.model[0], 'rb') as f:
            output['model'] = pickle.load(f)

        if args.verbose:
            print('done.')
    elif None not in (args.data, args.sim, args.grl):
        if args.verbose:
            print('Loading data...', end='', flush=True)

        output['data'] = numpy_multifile(args.data)

        if args.verbose:
            print('done.\nLoading sim...', end='', flush=True)

        output['sim'] = np.load(args.sim[0])

        if args.verbose:
            print('done.\nLoading GRL...', end='', flush=True)

        output['grl'] = numpy_multifile(args.grl)

        if args.verbose:
            print('done.')
    else:
        raise RuntimeError('You must either give a model or data/sim/grl')

    if args.mpl_backend is not None:
        mpl.use(args.mpl_backend[0])
        output['plot_inline'] = True

    output['outdir'] = args.outdir[0]
    if output['outdir'][-1] != '/':
        output['outdir'] += '/'

    output['verbose'] = args.verbose

    return output

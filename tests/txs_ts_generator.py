"""Docstring"""

import itertools
import argparse
import pickle
import numpy as np
import numpy.lib.recfunctions as rf

from context import mla


def main() -> None:
    """Docstring"""
    args = parse_args()
    analysis = load_analysis(args)

    trial = mla.analysis.produce_trial(
        analysis,
        flux_norm=args.flux_norm,
        verbose=args.verbose,
    )

    test_params = generate_params(args, trial, analysis)
    min_results = minimize_ts(args, trial, analysis, test_params)
    save(args, trial, min_results)


def parse_args() -> argparse.Namespace:
    """Docstring"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'analysis_pickle',
        metavar='A',
        help='The pickled TXS analysis object file location',
    )
    parser.add_argument(
        '-f', '--flux-norm',
        type=float,
        default=0,
        help='',
    )
    parser.add_argument(
        '-b', '--box-size',
        type=float,
        default=2,
        help='Size of box to cut events around source in degrees (default = 2)',
    )
    parser.add_argument(
        '-s', '--signalness',
        type=float,
        default=0.1,
        help='spatial signalness threshold for events (default = 0.1)',
    )
    parser.add_argument(
        '-m', '--to-minimize',
        nargs='*',
        default=None,
        help='list of parameters to minimize',
    )
    parser.add_argument(
        '-o', '--outdir',
        default='./',
        help='',
    )
    parser.add_argument(
        '-p', '--prefix',
        default='',
        help=''
    )
    parser.add_argument(
        '-v', '--verbose',
        default=False,
        action='store_const',
        const=True,
        help='',
    )

    args = parser.parse_args()

    if args.outdir[-1] != '/':
        args.outdir += '/'

    return args


def load_analysis(args: argparse.Namespace) -> mla.Analysis:
    """Docstring"""
    if args.verbose:
        print(
            f'Loading analysis file: {args.analysis_pickle}...',
            end='',
            flush=True,
        )

    with open(args.analysis_pickle, 'rb') as f:
        analysis = pickle.load(f)

    if args.verbose:
        print('done.')

    return analysis


def generate_params(
    args: argparse.Namespace,
    trial: np.ndarray,
    analysis: mla.Analysis,
) -> np.ndarray:
    """Docstring"""
    if args.verbose:
        print('generating time pairs...', end='', flush=True)

    in_ra_band = np.abs(
        analysis.source.ra - trial['ra']
    ) < np.radians(args.box_size)

    in_dec_band = np.abs(
        analysis.source.dec - trial['dec']
    ) < np.radians(args.box_size)

    signalness = analysis.model.signal_spatial_pdf(
        analysis.source,
        trial,
    ) / analysis.model.background_spatial_pdf(trial)

    valid_events = trial[
        (in_dec_band & in_ra_band) & (signalness > args.signalness)
    ]

    pairs = np.sort(
        np.array(list(itertools.combinations(valid_events['time'], 2))),
        axis=1,
    )

    params = mla.generate_params(
        start=pairs.T[0],
        length=np.diff(pairs, axis=1).T[0]
    )

    if args.verbose:
        print(f'done.\nnumber of pairs: {len(pairs)}')

    return params


def minimize_ts(
    args: argparse.Namespace,
    trial: np.ndarray,
    analysis: mla.Analysis,
    test_params: np.ndarray,
) -> np.ndarray:
    """Docstring"""
    min_results = mla.minimize_ts(
        analysis=analysis,
        events=trial,
        test_params=test_params,
        to_fit=args.to_minimize,
        verbose=args.verbose,
        as_array=True,
    )

    if args.verbose:
        print('adding time-adjusted test statistic...', end='', flush=True)

    transposed_results = np.ascontiguousarray(min_results.T)

    valid_livetime = np.array([
        analysis.model.contained_livetime(
            start=param_set['start'],
            stop=param_set['start'] + param_set['length'],
        )
        for param_set in transposed_results
    ])

    time_adjusted_ts = transposed_results['ts'] + 2 * np.log(
        valid_livetime / analysis.model.livetime
    )

    full_min_results = rf.append_fields(
        base=transposed_results,
        names=['livetime', 'time_adjusted_ts'],
        data=[valid_livetime, time_adjusted_ts],
    )

    if args.verbose:
        print('done.')

    return full_min_results


def save(
    args: argparse.Namespace,
    trial: np.ndarray,
    min_results: np.ndarray,
) -> None:
    """Docstring"""
    min_params_str = ''
    if args.to_minimize is not None:
        min_params_str = '_' + '_'.join(args.to_minimize)

    prefix = ''.join([
        args.outdir,
        args.prefix,
        f'txs_f{args.flux_norm:.2g}',
    ])

    trial_file = ''.join([prefix, '_trial.npy'])

    results_file = ''.join([
        prefix,
        f'_results_b{args.box_size:.2g}',
        f'_s{args.signalness:.2g}',
        min_params_str,
        '.npy',
    ])

    if args.verbose:
        print(f'saving trial file: {trial_file}...', end='', flush=True)

    np.save(trial_file, trial)

    if args.verbose:
        print(
            f'done.\nsaving results file: {results_file}...',
            end='',
            flush=None,
        )

    np.save(results_file, min_results.data)

    if args.verbose:
        print('done.')


if __name__ == '__main__':
    main()

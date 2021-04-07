"""Docstring"""

from typing import Tuple

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    """Docstring"""
    args = parse_args()
    trial, results = load(args)
    trial.sort(order='time')
    best_p_value = calc_best_p_value(args, trial, results)
    trimmed_trial, trimmed_best_p_value = trim_duplicates(trial, best_p_value)
    plot(args, trimmed_trial, trimmed_best_p_value)


def parse_args() -> argparse.Namespace:
    """Docstring"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'trial',
        metavar='T',
        help='',
    )
    parser.add_argument(
        'results',
        metavar='R',
        help='',
    )
    parser.add_argument(
        '-v', '--verbose',
        const=True,
        default=False,
        action='store_const',
        help='',
    )
    parser.add_argument(
        '-o', '--outfile',
        default='./txs_trial_plot.png',
        help='',
    )
    parser.add_argument(
        '-i', '--inline',
        const=True,
        default=False,
        action='store_const',
        help='',
    )

    return parser.parse_args()


def load(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Docstring"""
    if args.verbose:
        print(f'loading trial: {args.trial}...', end='', flush=True)

    trial = np.load(args.trial)

    if args.verbose:
        print(f'done.\nloading results: {args.results}...', end='', flush=True)

    results = np.load(args.results)

    if args.verbose:
        print('done.')

    return trial, results


def calc_best_p_value(
    args: argparse.Namespace,
    trial: np.ndarray,
    results: np.ndarray,
) -> np.ndarray:
    """Docstring"""

    if args.verbose:
        print('calculating best p-value for each event...', end='', flush=True)

    best_p_value = np.empty(len(trial))
    for i, event in enumerate(trial):
        diffs = event['time'] - results['start']
        valid_results = results[
            (diffs >= 0) & (diffs <= results['length'])
        ]

        if len(valid_results) > 0:
            best_ts = np.amax(valid_results['time_adjusted_ts'])
        else:
            best_ts = 0

        best_p_value[i] = np.sum(
            results['time_adjusted_ts'] >= best_ts
        ) / len(results)

    if args.verbose:
        print('done.')

    return best_p_value


def trim_duplicates(
    trial: np.ndarray,
    best_p_value: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Docstring"""
    first = best_p_value != np.roll(best_p_value, 1)
    last = best_p_value != np.roll(best_p_value, -1)
    trim_idx = (first | last) & (best_p_value != 0)
    return trial[trim_idx], best_p_value[trim_idx]


def plot(
    args: argparse.Namespace,
    trial: np.ndarray,
    best_p_value: np.ndarray,
) -> None:
    """Docstring"""
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['axes.linewidth'] = 5
    plt.rcParams['xtick.major.size'] = 15
    plt.rcParams['xtick.major.width'] = 5
    plt.rcParams['ytick.major.size'] = 15
    plt.rcParams['ytick.major.width'] = 5

    plt.figure(figsize=(10, 6))
    log_p_value = -np.log10(best_p_value)
    plt.plot(trial['time'][1:-1], log_p_value[1:-1], linewidth=5)
    plt.ylabel(r'$-log_{10}p$')
    plt.xticks([56293.70067, 56658.70067, 57023.70067], [2013, 2014, 2015])
    plt.grid(alpha=0.25, linewidth=3)

    if args.verbose:
        print(f'saving plot: {args.outfile}...', end='', flush=True)

    plt.savefig(args.outfile)

    if args.verbose:
        print('done.')

    if args.inline:
        plt.show()


if __name__ == '__main__':
    main()

"""Docstring"""

import sys
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from context import mla
import internal_args


def produce_and_minimize(
    pm_analysis: mla.Analysis,
    n_trials: int,
    **kwargs
) -> np.ndarray:
    """Docstring"""
    if args['verbose']:
        print(
            f'Producing and minimizing {n_trials} trials with ',
            end='',
            flush=True,
        )
        if 'flux_norm' in kwargs:
            print(
                f'flux_norm = {kwargs["flux_norm"]:.2g}...',
                end='',
                flush=True,
            )
        else:
            print(
                f'n_signal_observed = {kwargs["n_signal_observed"]}...',
                end='',
                flush=True,
            )

    arr = mla.produce_and_minimize(
        analysis=pm_analysis,
        n_trials=n_trials,
        as_array=True,
        **kwargs,
    )

    if args['verbose']:
        print('done.')

    return arr


def print_results(to_print: np.ndarray, column_width: int = 8) -> None:
    """Docstring"""
    header = ''.join(
        [name.center(column_width) for name in to_print.dtype.names],
    )

    lines = [header, *[
        ''.join([
            f'{x : .{min(4, column_width - 2)}g}'.ljust(column_width)
            for x in result
        ])
        for result in results
    ]]

    print('\n'.join(lines))


if __name__ == '__main__':
    # Parse command line arguments
    args = internal_args.parse(sys.argv)

    # Example source object
    source = mla.Source(
        name='TXS',
        ra=np.radians(77.3583),
        dec=np.radians(5.6931),
    )

    # Example time profile object
    uniform_profile = mla.UniformProfile(
        start=56224,
        length=158,
    )

    # Load or generate model based on command line arguments
    if args['model'] is not None:
        model = args['model']
    else:
        if args['verbose']:
            print('Generating model...', end='', flush=True)

        model = mla.I3EventModel(
            source=source,
            data=args['data'],
            sim=args['sim'],
            grl=args['grl'],
            gamma=-2,
            background_time_profile=copy.deepcopy(uniform_profile),
            signal_time_profile=copy.deepcopy(uniform_profile),
            sampling_width=np.radians(3),
            withinwindow=True,
        )
        model_file_loc = ''.join([args['outdir'], 'example_model.pkl'])

        if args['verbose']:
            print(
                f'done.\nSaving model to {model_file_loc}...',
                end='',
                flush=True,
            )

        with open(model_file_loc, 'wb') as f:
            pickle.dump(model, f)

        if args['verbose']:
            print('done.')

    # Example test statistic object
    ts = mla.LLHTestStatistic([
        mla.SpatialTerm(),
        mla.TimeTerm(
            background_time_profile=copy.deepcopy(uniform_profile),
            signal_time_profile=copy.deepcopy(uniform_profile),
        ),
        mla.I3EnergyTerm(gamma=-2),
    ])

    # Put it all together into an analysis object
    analysis = mla.Analysis(
        model=model,
        test_statistic=ts,
        source=source,
    )

    # Produce a trial
    if args['verbose']:
        print('Producing single trial...', end='', flush=True)

    trial = mla.produce_trial(
        analysis=analysis,
        n_signal_observed=100,
    )

    plot_file_loc = ''.join([args['outdir'], 'example_trial.png'])
    if args['verbose']:
        print(f'done.\nSaving plot to {plot_file_loc}...', end='', flush=True)

    # Plot the trial events
    plt.scatter(trial['time'], trial['logE'])
    plt.xlabel('time (MJD)')
    plt.ylabel(r'$log_{10}E$')
    plt.savefig(plot_file_loc)

    if args['verbose']:
        print('done.')
    if args['plot_inline']:
        plt.show()

    # Generate parameter array for minimizing
    test_params = mla.generate_params(
        gamma=-2,
        ns=90,
    )
    bounds = [(-4, -1), (0, 150)]

    # Minimize the test statistic using test_params
    if args['verbose']:
        print('Minimizing a single trial...', end='', flush=True)

    results = mla.minimize_ts(
        analysis=analysis,
        events=trial,
        bounds=bounds,
        test_params=test_params,
        as_array=True,
    )

    if args['verbose']:
        print('done.')

    print('Single trial minimization results:')
    print_results(results)

    # Produce 10 trials and minimize all at once
    if args['verbose']:
        print('Producing and minimizing 10 trials...', end='', flush=True)

    results = mla.produce_and_minimize(
        analysis=analysis,
        n_trials=10,
        n_signal_observed=100,
        bounds=bounds,
        test_params=test_params,
        as_array=True,
    )

    if args['verbose']:
        print('done.')

    print('Multiple trial minimization results:')
    print_results(results)

    # Produce and minimize 100 trials, then plot the ts distribution
    best_fit_arr = produce_and_minimize(
        pm_analysis=analysis,
        n_trials=100,
        n_signal_observed=100,
        bounds=bounds,
        test_params=test_params,
    )

    hist_file_loc = ''.join([args['outdir'], 'example_ts_dist.png'])
    if args['verbose']:
        print(f'Saving histogram to {hist_file_loc}...', end='', flush=True)

    plt.hist(best_fit_arr['ts'], histtype='step')
    plt.xlabel('Test Statistic')
    plt.ylabel('Counts')
    plt.savefig(hist_file_loc)

    if args['verbose']:
        print('done.')
    if args['plot_inline']:
        plt.show()

    # Now do the same over a range of flux normalization values
    flux_norms = np.empty(6)
    flux_norms[0] = 0
    flux_norms[1:] = np.logspace(-12, -10.5, 5)
    hists_file_loc = ''.join([args['outdir'], 'example_ts_dists.png'])
    bin_edges = np.arange(0, 250, 10)

    for flux_norm in flux_norms:
        best_fit_arr = produce_and_minimize(
            pm_analysis=analysis,
            n_trials=100,
            bounds=bounds,
            test_params=test_params,
            flux_norm=flux_norm,
        )

        plt.hist(
            best_fit_arr['ts'],
            histtype='step',
            bins=bin_edges,
            label=f'flux norm = {flux_norm:.2g}'
        )

    if args['verbose']:
        print(f'Saving histograms to {hists_file_loc}...', end='', flush=True)

    plt.xlabel('Test Statistic')
    plt.ylabel('Counts')
    plt.savefig(hists_file_loc)
    plt.legend()

    if args['verbose']:
        print('done.')
    if args['plot_inline']:
        plt.show()

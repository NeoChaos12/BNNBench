import logging
import pandas as pd
import numpy as np
import scipy.stats as stats

_log = logging.getLogger(__name__)

def get_rank_across_models(df: pd.DataFrame, metric: str = 'minimum_observed_value', nsamples: int = 1000,
                           method: str = 'average') -> \
        pd.DataFrame:
    """
    For a metrics dataframe, generate a corresponding dataframe containing model ranks based on randomly sampled
    rng_offsets for the chosen metric. The index of such a dataframe must correspond to this exact sequence of levels:
    ['model', 'metric', 'rng_offset', 'iteration'].
    :param df: pandas.DataFrame
        The DataFrame object containing the relevant metrics data.
    :param metric: string
        The name of the metric which is to be used for rank comparisons.
    :param nsamples: int
        The number of times each model's 'rng_offset' value is sampled.
    :param method: string
        The method used to generate the ranking, consult pandas.DataFrame.rank() for more details.
    :return: pandas.DataFrame
        A DataFrame object with the index levels ['model', 'sample_idx', 'iteration'] and the column labels
        ['rank', 'metric_value'] containing the respective rankings of each model and the metric values used to compute
        them, respectively, at each iteration, such that the new level 'sample_idx' denotes a randomly chosen sample of
        all iterations for each model.
    """

    known_names = ['model', 'metric', 'rng_offset', 'iteration']
    assert df.index.names == known_names, f"The input DataFrame must have this exact sequence of index names: " \
                                          f"{known_names}\nInstead, the given DataFrame's index was: {df.index.names}"
    metric_df = df.xs((slice(None), metric))
    models = metric_df.index.unique('model')
    chosen_offsets = [None] * len(models)
    for i, m in enumerate(models):
        tmp_df: pd.DataFrame = metric_df.xs(m, drop_level=False).unstack(level='iteration')
        choice = tmp_df.sample(nsamples, replace=True)
        choice['sample_idx'] = list(range(nsamples))
        chosen_offsets[i] = choice.set_index('sample_idx', append=True).reset_index(level='rng_offset', drop=True)

    new_df = pd.concat(chosen_offsets, axis=0).unstack(level='model').stack('iteration')
    rank_df = new_df.rank(axis=1, method=method).stack('model').rename(columns={'metric_value': 'rank'})
    rank_df = rank_df.combine_first(new_df.stack('model'))

    print("Finished sampling the dataframe.")
    return rank_df.reorder_levels(['model', 'sample_idx', 'iteration'])


def calculate_overhead(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe which has at least the levels ['model', 'metric', 'rng_offset', 'iteration'] as the final 4
    levels of its index, return a dataframe containing the estimated time taken for each iteration to generate a new
    data point to evaluate, including model re-training time, calculated by subtracting the time taken to evaluate the
    benchmark from the total time that passed between every two iterations.
    :param df: pandas.DataFrame
        The dataframe object containing metric data.
    :return: pandas.DataFrame
        A DataFrame object which has all the metric information and the same index as the object given as input along
        with an addition to the level 'metric' - 'overhead'. Note that the lowest value of 'iteration' (usually 0)
        denotes end of model initialization. Since no overhead can be calculated for this value, it will be be set to 0.
    """

    # Iteration 0 corresponds to the end of initialization.
    end_times = df.xs('time', level='metric', drop_level=True).unstack('iteration')
    iters = end_times.columns.unique('iteration')
    start_times = end_times.drop(iters.max(), axis=1, level='iteration')
    start_times = start_times.rename(lambda x: x+1, axis=1, level='iteration')
    # end_times = end_times.drop(iters.min(), axis=1, level='iteration')
    ind = start_times.columns.unique(0).values[0], iters.min()
    obj_eval_durations = df.xs('evaluation_duration', level='metric', drop_level=True).unstack('iteration')
    start_times[ind] = end_times[ind] - obj_eval_durations[ind]
    overhead = end_times - start_times - obj_eval_durations
    # start_times = pd.DataFrame()
    overhead: pd.DataFrame = overhead.stack('iteration').assign(metric='overhead')
    overhead = overhead.set_index('metric', append=True).reorder_levels(df.index.names)
    return df.combine_first(overhead)

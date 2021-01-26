import logging
import pandas as pd
import numpy as np
import scipy.stats as stats

_log = logging.getLogger(__name__)

def get_rank_across_models(df: pd.DataFrame, metric: str = 'minimum_observed_value', nsamples: int = 1000) -> \
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
    :return: pandas.DataFrame
        A DataFrame object with the index levels ['sample_idx', 'model', 'iteration'] and the column label 'rank'
        containing the respective rankings of each model at each iteration, such that the new level 'sample_idx'
        denotes a randomly chosen sample of all iterations for each model.
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

    new_df = pd.concat(chosen_offsets, axis=0)
    rank_df = new_df.unstack(level='model').stack('iteration').rank(axis=1).stack('model')

    print("Finished sampling the dataframe.")
    return rank_df.reorder_levels(['sample_idx', 'model', 'iteration']).rename(columns={'metric_value': 'rank'})

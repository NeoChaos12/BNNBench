import logging
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Sequence, Optional, Union, Any
from bnnbench.utils import constants as C

_log = logging.getLogger(__name__)


def mask_multiindex(df: pd.DataFrame, key: Sequence[Any], level: Union[Sequence[int], Sequence[str]], exclude=False) \
        -> pd.DataFrame:
    """ Given a key and a corresponding sequence of levels for a dataframe with a MultiIndex Index, return a view on
    the DataFrame that includes only the indices containing the given key (default). If exclude is True, all such
    indices are excluded from the view instead. """

    assert len(key) == len(level), "The number of levels do not correspond to the length of the key to be excluded."
    mask = np.ones_like(df.index.values).astype(bool)
    for k, l in zip(key, level):
        mask = mask * df.index.get_level_values(l).isin([k], level=l)

    if exclude:
        mask = np.invert(mask)

    return df[mask]


def normalize_scale(df: pd.DataFrame, level: Union[Sequence[int], Sequence[str]] = None):
    """ Reduce data in the given dataframe to a linear scale of [0.0, 1.0] using the min/max values across the
    specified level(s). """

    original_order = df.index.names
    shift_levels = original_order.difference(level)
    df = df.reorder_levels(level + shift_levels)

    if len(shift_levels) > 0:
        df = df.unstack(shift_levels)

    maxval: pd.Series = df.max(axis=1)
    minval: pd.Series = df.min(axis=1)
    scale: pd.Series = maxval - minval
    df: pd.DataFrame
    df = df.sub(minval, axis=0)
    df = df.div(scale, axis=0)

    if len(shift_levels) > 0:
        df = df.stack(shift_levels)

    df = df.reorder_levels(original_order)
    return df

def super_sample(df: pd.DataFrame, level: str = 'rng_offset', new_level='sample_idx', nsamples: int = 1000,
                 rng: int = 1) -> pd.DataFrame:
    """
    For a given dataframe containing raw metric values, super-sample the experiment data from the given index level and
    return the resultant Dataframe. All index levels that occur before 'level' in the sequence df.index.names will be
    iterated over. 'level' will be replaced by 'new_level'.
    :param df: pandas.DataFrame
        The dataframe object containing the data to be super-sampled
    :param level: str
        The index level to be super-sampled.
    :param new_level: str
        The name of the new level that will replace 'level', keeping track of the sequence in which samples were drawn.
    :param nsamples: int
        The number of samples to draw from level.
    :param rng: int
        A random number seed to ensure repeatable results.
    :return: pandas.DataFrame
        The dataframe containing super-sampled values.
    """

    level_index = df.index.names.index(level)
    extra_levels = df.index.names[:level_index]
    shift_levels = df.index.names[level_index+1:]
    if len(shift_levels) > 0:
        # There were levels in the index after the pivot 'level'
        df = df.unstack(shift_levels)

    if len(extra_levels) > 0:
        iter_values = pd.MultiIndex.from_product([df.index.unique(e) for e in extra_levels], names=extra_levels).values
    else:
        iter_values = [None]

    final_levels = extra_levels + [new_level]
    final_df = None
    for v in iter_values:
        if v != None:
            tmp = df.xs(v, level=extra_levels)
            if tmp.empty:
                # Accounts for cases where some expected unique indices are not present.
                _log.debug("Skipping unknown key %s." % str(v))
                continue
            new_index_values = [*v, list(range(nsamples))]
        else:
            tmp = df
            new_index_values = [list(range(nsamples))]
        try:
            super_df = tmp.sample(nsamples, replace=True, random_state=rng)
        except ValueError as e:
            print("boo")
            raise e
        if isinstance(super_df, pd.Series):
            super_df: pd.Series
            super_df = super_df.to_frame()
        super_df: pd.DataFrame
        super_df = super_df.assign(**dict(zip(final_levels, new_index_values)))
        super_df = super_df.set_index(final_levels, append=True).reset_index(level, drop=True)
        if final_df is None:
            final_df = super_df
        else:
            final_df = final_df.combine_first(super_df)

    if len(shift_levels) > 0:
        # There were levels in the index after the pivot 'level'
        final_df = final_df.stack(shift_levels)

    return final_df


def get_ranks(df: pd.DataFrame, across: str = 'model', method: str = 'average') -> pd.DataFrame:
    """
    For a given dataframe, generate a comparison of ranks across the specified level of the index. Returns a DataFrame
    with column 'rank'. The input DataFrame should have exactly one column, which will be replaced by 'rank'.

    :param df: pandas.DataFrame
        The DataFrame object containing the relevant metrics data.
    :param across: string
        The name of the index level across which rank is calculated.
    :param method: string
        The method used to generate the ranking, consult pandas.DataFrame.rank() for more details.
    :return: pandas.DataFrame
        The dataframe with rank data.
    """

    original_order = df.index.names
    df = df.unstack(across)
    df = df.rank(axis=1, method=method)
    df = df.stack(across)
    df = df.reorder_levels(original_order)
    return df


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
    end_times = df.xs('time', level='metric', drop_level=True)
    end_times = end_times.unstack('iteration')
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
    ret = df.combine_first(overhead)
    return ret


def perform_tsne(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Given a runhistory dataframe, generates TSNE embeddings in 2 dimensions for the data and returns the embedded
    data as a dataframe with the same index as the runhistory dataframe.

    The DataFrame itself should conform to these conditions:
    Row Index: Should be the Multi-Index with names defined in bnnbench.utils.constants.fixed_runhistory_row_index_labels,
    such that all values up to and including index "0" of the level "iteration" correspond to random samples and will
    be excluded from the t-SNE projection. Including these samples would pollute the embedding since they will attach
    an extremely high probability score to the random samples, and we are mostly only interested in the differences
    between the model generated samples. Therefore, all such samples are excluded at this stage itself rather than in
    the plotting stage. Also excluded are NaN values.
    Column Index: Homogenous in the column names i.e. include only the index level
    BenchmarkData.runhistory_base_col_name. Correspondingly, the returned dataframe will have precisely 3 column
    labels: "dim1", "dim2", and "objective_value", while the index level will be only "tsne_data".

    :param data: pandas.DataFrame
        The runhistory dataframe.
    :param n_components: int
        The number of components of the embedded space. Default is 2.
    """

    assert data.columns.nlevels == 1 and data.columns.names == (C.runhistory_data_level_name,), \
        f"The DataFrame 'data' should have a 1-level column index containing only the level name " \
        f"{C.runhistory_data_level_name}, was instead {data.columns.names} containing {data.columns.nlevels} levels."

    from sklearn.manifold import TSNE
    config_dims = data.columns.drop(C.y_value_label)
    # Get rid of random samples
    configs = data.xs(np.s_[1:], level=C.fixed_runhistory_row_index_labels[-1], drop_level=False)
    # Get rid of NaN values
    configs = configs[configs.notna().any(axis=1)]
    tsne = TSNE(n_components=n_components, n_jobs=1)
    # Perform t-SNE transformation on only the x-values
    tsne_data = tsne.fit_transform(configs.loc[pd.IndexSlice[:], config_dims].to_numpy())
    # Append y-values to configuration embeddings
    y_values = configs.loc[pd.IndexSlice[:], C.y_value_label]
    if tsne_data.shape[0] != y_values.shape[0]:
        raise RuntimeError("There is a mismatch in the number of data points mapped by t-SNE and the number of data "
                           "points expected.")
    tsne_data = np.concatenate((tsne_data, y_values.to_numpy().reshape(-1, 1)), axis=1)
    # Re-package the t-SNE embeddings into a DataFrame
    tsne_cols = pd.Index(data=C.tsne_data_col_labels, name=C.tsne_data_level_name)
    tsne_df = pd.DataFrame(data=tsne_data, index=configs.index, columns=tsne_cols)

    return tsne_df

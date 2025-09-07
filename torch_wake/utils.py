"""A bunch of useful stuff"""
import logging
from typing import List, Literal, Optional, Union

import pandas as pd
from pandas import DataFrame

# import xarray as xr
from xarray import DataArray, Dataset


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def drop_nondim_coords(
    ds: [DataArray, Dataset],
    to_keep: Optional[Union[str, List[str]]] = None,
) -> Union[DataArray, Dataset]:
    """
    Given a dataarray or dataset, drop all coordinates that are not \
        dimension of the object.

    Parameters
    ----------
    ds: DataArray | Dataset
        Input data array or dataset.
    to_keep: str | List[str] | None
        (Optional) A string or list of strings defining the coordinate(s) \
        that should be included in the returned object, event though they are \
        not dimensions.

    Returns
    -------
        DataArray | Dataset
    """
    to_keep = [[], to_keep][to_keep is not None]
    to_keep = [[to_keep], to_keep][isinstance(to_keep, list)]

    ds = ds.drop([c for c in ds.coords if c not in to_keep])
    return ds


def rename_index(
    df: DataFrame,
    name: str,
    axis: Literal['row', 'column'] = 'column'
) -> DataFrame:
    """
    Given a data frame, rename the selected index (row or columns) to \
        the given name.

    Parameters
    ----------
    df: DataFrame
        Input data frame
    name: str
        A string defininf the new index name.
    axis: 'row' | 'column'
        An option defining the which on the rows or columns index (default) \
        should be renamed.

    Returns
    -------
        DataFrame
    """
    # Check
    allowed = ['row', 'column']
    if axis not in allowed:
        msg = 'Invalid value for parameter axis. It must be one of: ' \
              f'{allowed}.\n But received: {axis}.\n Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    ind = {'row': df.index, 'column': df.columns}[axis]
    ind = pd.Index(ind, name=name)

    return df

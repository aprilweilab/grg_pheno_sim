"""
This file contains some validity checks for datatypes.
=======
"""

import numbers
import operator
import pandas as pd


def check_type(input, id, data_type):
    """
    This function checks if the input named `id` is of the required instance
    `data_type` and raises a type error with an associated message if not.

    """
    if isinstance(input, data_type):
        return input
    else:
        raise TypeError(f"{id} must be a {data_type} instance")


def check_val(input, id, minimum=None, inclusive=False):
    """
    This function checks that the input is numerical and greater than the minimum.
    """
    if not isinstance(input, numbers.Real):
        raise TypeError(f"{id} must be numeric")
    val = float(input)
    if minimum is not None and not inclusive and val <= minimum:
        raise ValueError(f"{id} must be a number greater than {minimum}")
    elif minimum is not None and inclusive and val < minimum:
        raise ValueError(f"{id} must be a number not less than {minimum}")
    return val


def check_int(input, id, minimum=None):
    """
    This function checks that the input is an integer and greater than the minimum.
    """
    try:
        input = operator.index(input)
    except TypeError:
        raise TypeError(f"{id} must be an integer") from None
    if minimum is not None and input < minimum:
        raise ValueError(
            f"{id} must be an integer not less " f"than {minimum}"
        ) from None
    return input


def check_pd_df(dataframe, df_id):
    """
    This function checks whether `dataframe` is a pandas dataframe or not.
    """

    df = check_type(dataframe, df_id, pd.DataFrame)

    return df

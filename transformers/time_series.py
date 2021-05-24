
from datetime import datetime
from typing import Optional, Union, List

import dateinfer
from pandas.core.dtypes.common import is_integer_dtype, is_datetime64_any_dtype
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np


def standardize_date_offset(date_offset_unit: Union[pd.DateOffset, dict, str]) -> pd.DateOffset:
    """
    Constructor for DateOffset from strings like "seconds", dicts like {"seconds": 2}, or DateOffset, into DateOffset
    Refer to: https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html
    :param date_offset_unit:
        May be a pd.DateOffset
            EX. date_offset_unit=pd.DateOffset(1, seconds=1)
        May also be a dictionary of offsets. NOTE: if >1 key, then the resampler will fail
            EX. date_offset_unit={'months': 3, 'years': 1}
        May also be a single unit
            EX. date_offset_unit='years'
    :return:
    """
    if isinstance(date_offset_unit, str):
        date_offset_unit = {date_offset_unit: 1}
    if isinstance(date_offset_unit, dict):
        date_offset_unit = pd.DateOffset(1, **date_offset_unit)
    if not isinstance(date_offset_unit, pd.DateOffset):
        raise ValueError("date_offset_unit must be one of pd.DateOffset, dict, or str")
    return date_offset_unit


def format_dataframe_date_index(
        dataframe: pd.DataFrame,
        order_column: str, *,
        indexes: List[str] = None,
        date_format: Optional[str] = None,
        date_offset_unit: Union[pd.DateOffset, dict, str, None] = None,
        date_offset_start: Union[datetime, str, None] = None):
    """
    Format dataframe to have a pd.DatetimeIndex from order_column. Ensure is sorted.

    :param dataframe: arbitrarily indexed dataframe. the index will be lost
    :param order_column: column name to turn into date index
    :param indexes: names of index columns to preserve as-is
    :param date_format: date format string in strptime format
    :param date_offset_unit: if truthy, treat order_column as an integer time offset.
        This arg is a date offset that is applied `order_column` number of times against date_offset_start.
        Standardized by standardize_date_offset.
    :param date_offset_start: initial datetime to apply offsets to

    :return Optional[pd.DateOffset] observation frequency
    """

    if not isinstance(order_column, str):
        raise ValueError('order_column must be a str')

    # when order column does not exist, create an incrementing column
    if order_column not in dataframe.columns:
        # do nothing if dataframe is already formatted correctly
        if isinstance(dataframe.index, pd.DatetimeIndex):
            dataframe.sort_index(inplace=True)
            return dataframe.index.freq

        dataframe[order_column] = np.arange(len(dataframe))
        if date_offset_unit is None:
            date_offset_unit = 'seconds'

    if date_offset_unit:
        if not is_integer_dtype(dataframe[order_column]):
            raise ValueError("order_column must be integral if date_offset is provided")

        date_offset_unit = standardize_date_offset(date_offset_unit)

        # standardize date_offset_start
        if date_offset_start is None:
            date_offset_start = datetime.utcfromtimestamp(0)
        if not isinstance(date_offset_start, datetime):
            date_offset_start = datetime.strptime(date_offset_start, format=date_format)

        # create datetime index from date offsets
        order_data = dataframe[order_column].apply(lambda x: date_offset_start + x * date_offset_unit)

    elif is_datetime64_any_dtype(dataframe[order_column]):
        order_data = dataframe[order_column]

    else:
        if date_format is None:
            raise ValueError("date_format must be known")
        # create datetime index by parsing the order column with given format
        order_data = pd.to_datetime(dataframe[order_column], format=date_format)

    if indexes is not None:
        indexes_data = dataframe[indexes]

    dataframe[order_column] = order_data
    dataframe.set_index(order_column, inplace=True)

    if indexes is not None:
        dataframe[indexes] = indexes_data

    dataframe.sort_index(inplace=True)

    return date_offset_unit


def resample_dataframe_date_index(
        dataframe: pd.DataFrame,
        resample_date_offset_unit: pd.DateOffset):
    """
    Ensure that dataframe is regularly spaced. May contain null values for missing datetimes
    :param dataframe: sorted non-na dataframe indexed by datetimes
    :param resample_date_offset_unit: offset between observations in the returned dataframe
    :return dataframe with df.index.freq equal to appropriate alias representing resample_date_offset_unit
    """
    assert isinstance(dataframe.index, pd.DatetimeIndex)
    assert resample_date_offset_unit is not None

    inferred_date_offset = pd.tseries.frequencies.to_offset(dataframe.index.inferred_freq)
    # if time series is regular and freq happens to match
    if resample_date_offset_unit == inferred_date_offset:
        dataframe.index.freq = dataframe.index.inferred_freq
        return dataframe

    # workaround for https://github.com/pandas-dev/pandas/issues/31697
    _aliases = {
        'years': 'Y',
        'months': 'MS',
        'weeks': 'W',
        'days': 'D',
        'hours': 'H',
        'minutes': 'T',
        'seconds': 'S',
        'microseconds': 'U',
        'nanoseconds': 'N',
    }
    if isinstance(resample_date_offset_unit, pd.DateOffset):
        if len(resample_date_offset_unit.kwds) != 1:
            raise ValueError(f"date_offset_unit must have one alias: {resample_date_offset_unit.kwds}")
        for kwd in resample_date_offset_unit.kwds:
            resample_date_offset_unit = f'{resample_date_offset_unit.n * resample_date_offset_unit.kwds[kwd]}{_aliases[kwd]}'

    return dataframe.resample(resample_date_offset_unit).interpolate(method='time')


def get_min_freq(series):
    """
    Infer observation frequency from data
    :param series: https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.infer_freq.html
    :return observation frequency
    """
    if series is None:
        return None

    # infer frequency from triplets of records
    candidate_frequencies = set()

    last_trio = len(series) - 3
    num_samples = min(last_trio, 100)
    for i in np.round(np.linspace(0, last_trio, num_samples)).astype(int):
        candidate_frequency = pd.infer_freq(series[i:i + 3])
        if candidate_frequency:
            candidate_frequencies.add(candidate_frequency)

    # if data has no trio of evenly spaced records
    if not candidate_frequencies:
        return

    # approximately select shortest inferred frequency
    return min(candidate_frequencies, key=approx_seconds)


def approx_seconds(offset: pd.DateOffset) -> float:
    """
    Approximate the number of seconds in the duration of a DateOffset
    :param offset: pandas DateOffset instance
    :return seconds
    """
    if not offset:
        return

    try:
        offset = pd.tseries.frequencies.to_offset(offset)
        if offset:
            return offset.nanos / 1E9
    except ValueError:
        pass

    # find the elapsed time for one application of the DateOffset
    date = offset.rollback(datetime.now())
    return ((date + offset) - date).total_seconds()


class TemporalPreprocessor(TransformerMixin, BaseEstimator):
    """
    Sklearn data preprocessor for time series data.
    1. standardize the order_column to a DatetimeIndex.
       if order_column not exists:
        - insert incrementing "time index offset column"
       if order_column is an integral "time index offset column":
        - translates "time index offset column" into datetimes via-- start_datetime + i * DateOffset
        - DateOffset defaults to one second if not provided
       else:
        - detects date format if not exists
        - parses order column into DatetimeIndex
    2. ensures index column is sorted
    3. optionally resamples to make a regular time series
        - if a resample time offset unit not given, detects the natural time differences in the data (min offset)
        - either downsamples/upsamples/imputes missing time points to create a regular time series at time offset units
    4. TODO: detrending, one of {linear, bandpass, compensate for error change, etc.}
    5. TODO: seasonality adjustment, difference by value at time offset, must be invertible
    """
    def __init__(self, order_column, *,
                 indexes=None,
                 date_format=None,
                 input_date_offset_unit=None, date_offset_start=None,
                 resample=False, resample_date_offset_unit=None,
                 decompose=False):
        self.order_column = order_column
        self.indexes = indexes
        self.date_format = date_format
        self.input_date_offset_unit = input_date_offset_unit
        self.date_offset_start = date_offset_start
        self.resample = resample
        self.resample_date_offset_unit = resample_date_offset_unit
        self.decompose = decompose

        self.date_offset_end = None

    def fit(self, X, y=None):
        if y is not None:
            raise ValueError("y must be None")

        # set date_offset_unit to seconds if we will be building a new ordering
        if self.order_column not in X.columns:
            if self.input_date_offset_unit is None:
                self.input_date_offset_unit = pd.DateOffset(1, seconds=1)

        # if we will be parsing directly to datetimes, ensure date_format and resample_date_offset_unit is known
        if self.input_date_offset_unit is None:
            if self.order_column not in X.columns:
                raise ValueError("order_column is not present in X, and no date_offset_unit is given")

            # need to know date_format for parsing unknown data
            if self.date_format is None:
                self.date_format = dateinfer.infer(X[self.order_column].astype('str'))

            ordering = X[self.order_column]
            if not is_datetime64_any_dtype(ordering):
                ordering = pd.to_datetime(ordering, format=self.date_format)
            ordering = ordering.sort_values()

            self.date_offset_end = ordering.iloc[-1]

            # both date_offset_units are unknown! we must infer the output freq
            if self.resample and self.resample_date_offset_unit is None:

                # estimated_unit = pd.tseries.frequencies.to_offset(ordering.inferred_freq)
                estimated_unit = pd.tseries.frequencies.to_offset(get_min_freq(ordering))

                # fall back to line-space if data is completely irregular
                if estimated_unit is None:
                    estimated_unit = pd.DateOffset(seconds=((ordering[-1] - ordering[0]) / len(ordering)).total_seconds())
                self.resample_date_offset_unit = estimated_unit

        return self

    def transform(self, X: pd.DataFrame, decompose=False):
        """

        :param X: dataset to format
        :param decompose: set to True to return trend, seasonality, noise components, if fitted with seasonal_decompose
        :return:
        """
        if decompose:
            raise NotImplementedError("decomposition is not implemented in TemporalFormatter")

        format_dataframe_date_index(
            X, self.order_column,
            indexes=self.indexes,
            date_format=self.date_format,
            date_offset_unit=self.input_date_offset_unit,
            date_offset_start=self.date_offset_start)

        if self.resample:
            resample_date_offset_unit = self.resample_date_offset_unit or self.input_date_offset_unit
            if self.indexes and resample_date_offset_unit != self.input_date_offset_unit:
                # drop indexes columns when resampling with a different time granularity
                # note that problem.indexes may be different from the datetime dataframe.index
                # ignore errors because sometimes problem.indexes have been consumed into the dataframe.index
                X.drop(self.indexes, errors='ignore', inplace=True)
            X = resample_dataframe_date_index(X, resample_date_offset_unit=resample_date_offset_unit)
        return X

    def inverse_transform(self, Y):
        # do... nothing
        return Y


class TemporalImputer(TransformerMixin):
    def fit(self, X, y=None):
        if y is not None:
            raise ValueError('y must be None')
        return self

    def transform(self, X, y=None):
        if y is not None:
            raise ValueError('y must be None')
        X.interpolate(method='linear', inplace=True)
        return X

    def inverse_transform(self, X, y=None):
        return X

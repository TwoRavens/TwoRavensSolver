import os
from dateutil import parser

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from datetime import datetime


def filter_args(arguments, intersect):
    return {k: arguments[k] for k in intersect if k in arguments}


def format_dataframe_time_index(dataframe, date=None, granularity_specification=None):
    """
    Ensure dataframe index is of type pd.DatetimeIndex on the date column
    @param dataframe: arbitrarily indexed dataframe
    @param date: optional column name to turn into date index
    @param granularity_specification: freq in d3m format
    """
    if date is None:
        date = 'ravensDateIndex'
        while date in dataframe:
            date += '_'

    # attempt to parse given date column
    if date in dataframe:
        try:
            dataframe[date] = pd.to_datetime(dataframe[date], infer_datetime_format=True)
            return resample_dataframe_time_index(
                dataframe=dataframe,
                date=date,
                freq=get_freq(granularity_specification=granularity_specification))
        except ValueError:
            pass

    # if there was a spec, but no valid date column to apply it to, then ignore it
    dataframe[date] = pd.date_range('1900-1-1', periods=len(dataframe), freq='D')
    return dataframe.set_index(date)


def resample_dataframe_time_index(dataframe, date, freq=None):
    """
    Creates a regular time series, optionally at the specified freq
    @param dataframe:
    @param date:
    @param freq:
    @return:
    """
    temporal_series = dataframe[date]
    estimated_freq = get_freq(series=temporal_series)

    # fall back to linspace if data is completely irregular
    if not estimated_freq:
        estimated_freq = (temporal_series[-1] - temporal_series[0]) / len(dataframe)

    # if time series is regular and freq happens to match
    if pd.infer_freq(temporal_series) and approx_seconds(freq) == approx_seconds(estimated_freq):
        return dataframe.set_index(date)

    freq = freq or estimated_freq

    dataframe = dataframe.set_index(date)
    dataframe_temp = dataframe.resample(freq).mean()

    numeric_columns = list(dataframe.select_dtypes(include=[np.number]).columns.values)
    categorical_columns = [i for i in dataframe.columns.values if i not in numeric_columns]

    for dropped_column in categorical_columns:
        dataframe_temp[dropped_column] = dataframe[dropped_column]

    dataframe_imputed = pd.DataFrame(ColumnTransformer(transformers=[
        ('numeric', SimpleImputer(strategy='median'), numeric_columns),
        ('categorical', SimpleImputer(strategy='most_frequent'), categorical_columns)
    ]).fit_transform(dataframe_temp), index=dataframe_temp.index, columns=dataframe_temp.columns)

    # no imputations on index column
    if 'd3mIndex' in dataframe_temp:
        dataframe_imputed['d3mIndex'] = dataframe_temp['d3mIndex']
    return dataframe_imputed


def get_freq(series=None, granularity_specification=None):
    """
    Infer observation frequency given d3m metadata or data
    @param granularity_specification: https://gitlab.com/datadrivendiscovery/data-supply/blob/4d67a8acee3fe5236900137a528bc48cf05731a3/schemas/datasetSchema.json#L101
    @param series: https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.infer_freq.html
    @return: observation frequency
    """
    d3m_granularity_units = {
        "seconds": "S",
        "minutes": "T",
        "days": "D",
        "weeks": "W",
        "years": "Y"
    }

    # attempt to build unit from user metadata
    if granularity_specification and granularity_specification['units'] in d3m_granularity_units:
        value = granularity_specification.get('value')
        return (str(value) if value else '') + d3m_granularity_units[granularity_specification['units']]

    if series is None:
        return None

    # infer frequency from every three-pair of records
    candidate_frequencies = set()
    for i in range(len(series) - 3):
        candidate_frequency = pd.infer_freq(series[i:i + 3])
        if candidate_frequency:
            candidate_frequencies.add(candidate_frequency)

    # if data has no trio of evenly spaced records
    if not candidate_frequencies:
        return

    # sort inferred frequency by approximate time durations, select shortest
    return sorted([(i, approx_seconds(i)) for i in candidate_frequencies], key=lambda x: x[1])[0][0]


def get_date(value, time_format=None):
    try:
        return datetime.strptime(value, time_format) if time_format else parser.parse(str(value))
    except (parser._parser.ParserError, ValueError):
        # ignore if could not be parsed
        pass

# otherwise take the shortest date offset
def approx_seconds(offset):
    """
    Attempt to approximate the number of seconds in the duration of a DateOffset
    @param offset: pandas DateOffset instance
    @return: float seconds
    """
    if not offset:
        return
    offset = pd.tseries.frequencies.to_offset(offset)
    try:
        if offset:
            return offset.nanos / 1E9
    except ValueError:
        pass

    date = datetime.now()
    return ((offset.rollback(date) - offset.rollforward(date)) * offset.n).total_seconds()


class Dataset(object):
    def __init__(self, input):
        if not input:
            raise ValueError('No input provided.')

        if 'resource_uri' not in input:
            raise ValueError('Invalid input: no resource_uri provided.')

        self.input = input

    def get_dataframe(self):
        options = {}

        if 'delimiter' in self.input:
            options['delimiter'] = self.input['delimiter']

        return pd.read_csv(self.get_resource_uri(), **options)

    def get_resource_uri(self):
        return self.input['resource_uri']

    def get_file_path(self):
        return os.path.join(*self.get_resource_uri().replace('file://', '').split('/'))

    def get_name(self):
        return self.input.get('name', self.input['resource_uri'])


def preprocess(dataframe, specification):

    X = specification['problem']['predictors']
    y = specification['problem']['targets'][0]
    nominal = [i for i in specification['problem'].get('categorical', []) if i in X]
    dataframe[nominal] = dataframe[nominal].astype(str)

    categorical_features = [i for i in set(nominal +
                            list(dataframe.select_dtypes(exclude=[np.number, "bool_", "object_"]).columns.values))
                            if i != y and i in X]

    # keep up to the 20 most frequent levels
    categories = [dataframe[col].value_counts()[:20].index.tolist() for col in categorical_features]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False))
    ])

    numerical_features = list(dataframe.select_dtypes(include=[np.number]))
    numerical_features = [i for i in X if i not in categorical_features and i in numerical_features]
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    stimulus = dataframe[X]
    stimulus = preprocessor.fit_transform(stimulus)

    return stimulus, preprocessor

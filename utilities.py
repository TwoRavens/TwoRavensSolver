import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def filter_args(arguments, intersect):
    return {k: arguments[k] for k in intersect if k in arguments}


def format_dataframe_time_index(dataframe, date=None):
    """
    Ensure dataframe index is of type pd.DatetimeIndex on the date column
    @param dataframe: arbitrarily indexed dataframe
    @param date: optional column name to turn into date index
    """
    log = []

    if (date is None or dataframe.index.name == date) and isinstance(dataframe.index, pd.DatetimeIndex):
        return format_dataframe_time_index(dataframe, date or dataframe.index.name), log

    if date is None:
        date = 'ravensDateIndex'
        while date in dataframe:
            date += '_'

    # attempt to parse given date column
    if date in dataframe:
        try:
            dataframe[date] = pd.to_datetime(dataframe[date], infer_datetime_format=True)
            dataframe = dataframe.set_index(date)
            dataframe = evenly_resample_time_series(dataframe)
            return dataframe, log
        except ValueError:
            log.append('date column provided, but could not be parsed')

    dataframe[date] = pd.date_range('1900-1-1', periods=len(dataframe), freq='D')
    log.append('equidistant date column added')
    return dataframe.set_index(date).dropna(), log


def get_freq(granularity_specification=None, dates=None):
    """
    Infer observation frequency given d3m metadata or data
    @param granularity_specification: https://gitlab.com/datadrivendiscovery/data-supply/blob/4d67a8acee3fe5236900137a528bc48cf05731a3/schemas/datasetSchema.json#L101
    @param dates: https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.infer_freq.html
    @return: observation frequency
    """
    if granularity_specification and 'units' in granularity_specification:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        unit = {
            "seconds": "S",
            "minutes": "T",
            "days": "D",
            "weeks": "W",
            "years": "Y",
            "unspecified": None
        }.get(granularity_specification['units'])
        if unit:
            value = granularity_specification.get('value')
            return (str(value) if value else '') + unit
    if dates is not None:
        return pd.infer_freq(dates)


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

    categorical_features = [i for i in specification['problem']['categorical'] if i != y and i in X]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_features = [i for i in X if i not in categorical_features]
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


def evenly_resample_time_series(dataframe):
    dataframe = dataframe.resample(
        (dataframe.index[-1] - dataframe.index[0]) / len(dataframe)
    ).mean()

    imputed = pd.DataFrame(
        SimpleImputer(missing_values=np.nan, strategy='mean')
            .fit_transform(dataframe))

    imputed.columns = dataframe.columns
    imputed.index = dataframe.index

    return imputed

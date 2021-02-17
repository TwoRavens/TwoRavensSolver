import os
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

pd.options.mode.chained_assignment = 'raise'

DEFAULT_INDEX = 'd3mIndex'


def filter_args(arguments, intersect):
    return {k: arguments[k] for k in intersect if k in arguments}


def format_dataframe_time_index(
        dataframe,
        order_column=None,
        time_format=None,
        start_dummy='1900-1-1'):
    """
    Ensure dataframe index is of type pd.DatetimeIndex on the date column
    :param dataframe: arbitrarily indexed dataframe
    :param order_column: optional column name to turn into date index
    :param start_dummy:
    :param time_format: if truthy, consider data is_date
    """

    # Sanity check
    if order_column is None and time_format:
        raise ValueError('time_format must be None when date is None')

    # Create dummy date index if date is not given
    if order_column is None:
        order_column = 'ravensDateIndex'
        while order_column in dataframe:
            order_column += '_'

    # Attempt to parse given date column
    if order_column in dataframe and time_format:
        try:
            # Attempt to parse the input string with given format
            dataframe[order_column] = dataframe[order_column].astype(str).apply(lambda x: get_date(x, time_format))
            dataframe.set_index(order_column, inplace=True)
            return dataframe
        except ValueError:
            print(f"Failed to parse order column with the given time_format: {time_format}")
            pass

        # Date format from the preprocessor is not available, or fails to parse
        try:
            dataframe[order_column] = dataframe[order_column].astype(str).apply(parser.parse)
            dataframe.set_index(order_column, inplace=True)
            return dataframe
        except ValueError:
            pass

    # if there was a spec, but no valid date column to apply it to, then ignore it
    dataframe[order_column] = pd.date_range(start_dummy, periods=len(dataframe), freq='D')
    dataframe.set_index(order_column, inplace=True)
    return dataframe


def resample_dataframe_time_index(dataframe, freq=None, index_name=DEFAULT_INDEX):
    """
    Creates a regular time series, optionally at the specified freq
    :param dataframe:
    :param freq:
    :param index_name:
    :return
    """
    temporal_series = dataframe.index
    estimated_freq = get_freq(series=temporal_series)
    # print(estimated_freq)

    # fall back to line-space if data is completely irregular
    if temporal_series is not None and not estimated_freq:
        estimated_freq = (temporal_series.iloc[-1] - temporal_series.iloc[0]) / len(dataframe)

    # if time series is regular and freq happens to match
    if pd.infer_freq(temporal_series) and approx_seconds(freq) == approx_seconds(estimated_freq):
        return dataframe

    freq = freq or estimated_freq

    dataframe_temp = dataframe.resample(freq).mean()

    numeric_columns = list(dataframe.select_dtypes(include=[np.number]).columns.values)
    categorical_columns = [i for i in dataframe.columns.values if i not in numeric_columns]

    for dropped_column in categorical_columns:
        dataframe_temp[dropped_column] = dataframe[dropped_column]

    # drop columns that are completely na
    dataframe_temp = dataframe_temp.dropna(how='all', axis=1)

    dataframe_imputed = pd.DataFrame(ColumnTransformer(transformers=[
        ('numeric', SimpleImputer(strategy='median'), numeric_columns),
        ('categorical', SimpleImputer(strategy='most_frequent'), categorical_columns)
    ]).fit_transform(dataframe_temp), index=dataframe_temp.index, columns=dataframe_temp.columns)

    # no imputations on index column
    if index_name in dataframe_temp:
        dataframe_imputed[index_name] = dataframe_temp[index_name]
    return dataframe_imputed


def get_freq(series=None, granularity_specification=None):
    """
    Infer observation frequency given d3m metadata or data
    :param granularity_specification: https://gitlab.com/datadrivendiscovery/data-supply/blob/4d67a8acee3fe5236900137a528bc48cf05731a3/schemas/datasetSchema.json#L101
    :param series: https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.infer_freq.html
    :return observation frequency
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

    # business_units = {
    #     # "SM": "M",
    #     "BM": "M",
    #     "CBM": "M",
    #     # "SMS": "MS",
    #     "BMS": "MS",
    #     "CBMS": "MS",
    #     "BQ": "Q",
    #     "BQS": "QS",
    #     "BA": "A",
    #     "BY": "Y",
    #     "BAS": "AS",
    #     "BYS": "YS",
    #     "BH": "H",
    # }
    # infer frequency from every three-pair of records
    candidate_frequencies = set()
    # TODO: spread out samples if series length longer than 100
    for i in range(min(len(series) - 3, 100)):
        candidate_frequency = pd.infer_freq(series[i:i + 3])
        if candidate_frequency:
            # for unit in business_units:
            #     candidate_frequency.replace(unit, business_units[unit])
            candidate_frequencies.add(candidate_frequency)

    # if data has no trio of evenly spaced records
    if not candidate_frequencies:
        return

    # approximately select shortest inferred frequency
    return min(candidate_frequencies, key=approx_seconds)


def get_date(value, time_format=None):
    try:
        return datetime.strptime(value, time_format) if time_format else parser.parse(str(value))
    except (parser._parser.ParserError, ValueError):
        pass

# otherwise take the shortest date offset
def approx_seconds(offset):
    """
    Attempt to approximate the number of seconds in the duration of a DateOffset
    :param offset: pandas DateOffset instance
    :return float seconds
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


def preprocess(dataframe, specification, X=None, y=None):
    assert len(specification['problem']['targets']) == 1

    X = X if X else specification['problem']['predictors']
    y = y if y else specification['problem']['targets'][0]

    nominal = [i for i in specification['problem'].get('categorical', []) if i in X]
    dataframe[nominal] = dataframe[nominal].astype(str)

    # list columns that must be numeric, according to the datatype
    categorical_dtype_features = list(dataframe.select_dtypes(exclude=[np.number, "bool_", "object_"]).columns.values)
    # union categorical dtype features with features the user has explicitly labeled as categorical
    categorical_features = [i for i in set(nominal + categorical_dtype_features)
                            # must not be a target, must be a predictor
                            if i != y and i in X]

    # print('preprocess X', X)
    # print('preprocess cate features')
    # print(categorical_features)

    # keep up to the 20 most frequent levels
    categories = [dataframe[col].value_counts()[:20].index.tolist() for col in categorical_features]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False))
    ])

    # numerical_features = dataframe.select_dtypes([np.number]).columns.values
    # print('dtypes', dataframe.dtypes)
    # print('dtype numeric', numerical_features)
    numerical_features = [i for i in X if i not in categorical_features
                          # and i in numerical_features
                          ]
    # print('preprocess numerical features')
    # print(numerical_features)
    # print(dataframe)

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

    set_column_transformer_inverse_transform(preprocessor, len(numerical_features))

    return stimulus, preprocessor


def split_time_series(dataframe, cross_section_names=None):
    """
    break a dataframe with cross sectional indicators into a dict of dataframes containing each treatment
    :param dataframe:
    :param cross_section_names: column names of cross sectional variables
    :return
    """

    # avoid unecessary data re-allocation
    if not cross_section_names:
        return {(): dataframe}

    others = [i for i in dataframe.columns.values if i not in cross_section_names]
    content = {label: data[others] for label, data in dataframe.groupby(cross_section_names)}
    for eachGroup in content:
        content[eachGroup].reset_index(drop=True, inplace=True)
    return content
    # return {label: data[others] for label, data in dataframe.groupby(cross_section_names)}


def set_column_transformer_inverse_transform(transformer, num_numerical_features):

    if not isinstance(transformer, ColumnTransformer):
        raise ValueError('expected ColumnTransformer when setting inverse_transform')
    if len(transformer.transformers) != 2:
        raise ValueError('ColumnTransformer inverse_transform hack only works with 2 transformers')

    def fixed_column_transformer_inverse_transform(data):

        print(transformer.transformers_[0][1].steps[1][1])
        numerical = pd.DataFrame(transformer.transformers_[0][1].steps[1][1].inverse_transform(data.iloc[:, :num_numerical_features]))
        # categorical = transformer.transformers[1][1].inverse_transform(data.iloc[:, num_numerical_features:])
        categorical = pd.DataFrame(data.iloc[:, num_numerical_features:])
        return pd.concat([numerical, categorical], axis=1)

    setattr(transformer, 'inverse_transform', fixed_column_transformer_inverse_transform)

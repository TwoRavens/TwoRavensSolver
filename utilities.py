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


def format_dataframe_time_index(dataframe, date=None, granularity_specification=None, freq=None, date_format=None):
    """
    Ensure dataframe index is of type pd.DatetimeIndex on the date column
    @param dataframe: arbitrarily indexed dataframe
    @param date: optional column name to turn into date index
    @param granularity_specification: freq in d3m format
    """

    # Sanity check
    if date is None and date_format:
        raise ValueError('date_format must be None when date is None')

    # Create dummy date index if date is not given
    if date is None:
        date = 'ravensDateIndex'
        while date in dataframe:
            date += '_'

    # Attempt to parse given date column
    if date in dataframe:
        # Date format is not available from the preprocessor
        if not date_format:
            try:
                dataframe[date] = dataframe[date].astype(str).apply(parser.parse)
                # not flexible enough
                return resample_dataframe_time_index(
                    dataframe=dataframe,
                    date=date,
                    freq=freq or get_freq(granularity_specification=granularity_specification))
            except ValueError:
                pass
        else:
            try:
                # Trying to parse the input string with given format
                dataframe[date] = dataframe[date].astype(str).apply(lambda x: get_date(x, date_format))
                return resample_dataframe_time_index(
                    dataframe=dataframe,
                    date=date,
                    freq=freq or get_freq(granularity_specification=granularity_specification))
            except ValueError:
                # Fall back to no format version
                try:
                    dataframe[date] = dataframe[date].astype(str).apply(parser.parse)
                    # not flexible enough
                    return resample_dataframe_time_index(
                        dataframe=dataframe,
                        date=date,
                        freq=freq or get_freq(granularity_specification=granularity_specification))
                except ValueError:
                    pass

    # if there was a spec, but no valid date column to apply it to, then ignore it
    dataframe[date] = pd.date_range('1900-1-1', periods=len(dataframe), freq='D')
    return dataframe.set_index(date)


# New function for general ordered dataframe
def format_dataframe_order_index(dataframe, order_column=None, is_date=True, date_format=None,
                                 granularity_specification=None, freq=None, start_dummy='1900-1-1'):
    """
    Ensure dataframe index is of type pd.DatetimeIndex (if is_date is True) on the date column
    Otherwise, the index is of type pd.RangeIndex -- Only numerical value is supported

    @param dataframe: arbitrarily indexed dataframe
    @param order_column: optional column name to turn into date index
    @param is_date: boolean flag that denotes the index type
    @param date_format: date format from 2ravens_preprocessor
    @param granularity_specification: freq in d3m format
    @param freq: frequency for the given column, optional
    @param start_dummy: the start date of dummy index,
    """

    # Sanity check
    if order_column is None and date_format:
        raise ValueError('date_format must be None when date is None')

    # order_column would never be None, based on the front-end implementation
    # if order_column is None or order_column == 'd3mIndex':
    #     order_column = 'ravensDateIndex'
    #     while order_column in dataframe:
    #         order_column += '_'

    # Attempt to parse given date column
    if order_column in dataframe:
        if is_date:
            # Try parse the dateTimeIndex using given format first
            if date_format:
                try:
                    dataframe[order_column] = dataframe[order_column].astype(str).apply(
                        lambda x: get_date(x, date_format))
                    return resample_dataframe_time_index(
                        dataframe=dataframe,
                        date=order_column,
                        freq=freq or get_freq(granularity_specification=granularity_specification)
                    ), None
                except ValueError:
                    pass

            # Guess the dateTimeIndex if format is not given, or parse failed with given format
            try:
                dataframe[order_column] = dataframe[order_column].astype(str).apply(parser.parse)
                return resample_dataframe_time_index(
                    dataframe=dataframe,
                    date=order_column,
                    freq=freq or get_freq(granularity_specification=granularity_specification)), None
            except ValueError:
                pass

    # if there was a spec, but no valid date column to apply it to, then ignore it
    # All parse branch failed, go for dummy dateIndex
    dummy_column = 'ravensDateIndex'
    while dummy_column in dataframe:
        dummy_column += '_'

    dummy_series = pd.date_range(start_dummy, periods=len(dataframe), freq='D')

    dataframe.insert(0, dummy_column, dummy_series)

    mapping_dic = {
        'start': [dataframe[dummy_column][0], dataframe[order_column][0]],
        'end': [dataframe[dummy_column][len(dataframe)-1], dataframe[order_column][len(dataframe)-1]],
        'index': dummy_column
    }

    # raise Exception(mapping_dic)

    dataframe = dataframe.set_index(dummy_column)

    return dataframe, mapping_dic
    #
    # if is_date:
    #     return dataframe, None
    # else:
    #     return dataframe, mapping_dic


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
    # print(estimated_freq)

    # fall back to line-space if data is completely irregular
    if temporal_series is not None and not estimated_freq:
        estimated_freq = (temporal_series.iloc[-1] - temporal_series.iloc[0]) / len(dataframe)

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

    # drop columns that are completely na
    dataframe_temp = dataframe_temp.dropna(how='all', axis=1)

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
    for i in range(len(series) - 3):
        candidate_frequency = pd.infer_freq(series[i:i + 3])
        if candidate_frequency:
            # for unit in business_units:
            #     candidate_frequency.replace(unit, business_units[unit])
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


def preprocess(dataframe, specification, X=None, y=None):

    X = X if X else specification['problem']['predictors']
    y = y if y else specification['problem']['targets'][0]
    nominal = [i for i in specification['problem'].get('categorical', []) if i in X]
    dataframe[nominal] = dataframe[nominal].astype(str)

    categorical_features = [i for i in set(nominal +
                            list(dataframe.select_dtypes(exclude=[np.number, "bool_", "object_"]).columns.values))
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

    return stimulus, preprocessor


def split_time_series(dataframe, cross_section_names=None):
    """
    break a dataframe with cross sectional indicators into a dict of dataframes containing each treatment
    @param dataframe:
    @param cross_section_names: column names of cross sectional variables
    @return:
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

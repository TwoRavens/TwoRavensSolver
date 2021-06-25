from datetime import datetime

import numpy as np
import pandas as pd

from tworaven_solver.transformers.general import ProblemPreprocessor
from tworaven_solver.transformers.time_series import TemporalPreprocessor, format_dataframe_date_index
from tworaven_solver.tests.shared import problems
from tworaven_solver.utilities import TrainSpecification, Dataset

import matplotlib.pyplot as plt


def test_idempotent():
    """preprocess should return an invertible pipeline"""
    train_spec = TrainSpecification(problems['appliance']['train_specification'])
    dataframe = Dataset(train_spec['input']).get_dataframe(usecols=train_spec.problem.predictors)

    preprocessor = ProblemPreprocessor(train_spec.problem, {})
    preprocessor.fit(dataframe)
    test_data = preprocessor.transform(dataframe)
    inverted = preprocessor.inverse_transform(test_data)

    assert np.array_equal(dataframe.to_numpy(), inverted.to_numpy())


def test_temporal_formatter_sunspots(plot=False):
    train_spec = TrainSpecification(problems['sunspots']['train_specification'])
    dataframe = Dataset(train_spec['input']).get_dataframe(usecols=train_spec.problem.train)

    formatter = TemporalPreprocessor(order_column=train_spec.problem.ordering)
    formatter.fit(dataframe)
    assert formatter.date_format == "%Y-%m"
    result1 = formatter.transform(dataframe)

    # check that data is parsed correctly
    assert result1.index[0] == datetime(1749, 1, 1)
    assert result1.index.name == 'year-month'
    assert isinstance(result1.index, pd.DatetimeIndex)

    formatter = TemporalPreprocessor(
        order_column=train_spec.problem.ordering,
        resample=True,
        resample_date_offset_unit=pd.DateOffset(months=1))
    formatter.fit(dataframe)
    result2 = formatter.transform(dataframe)

    print(result1)
    print(result2)
    if plot:
        plt.plot('sunspots', data=result1)
        plt.plot('sunspots', data=result2)
        plt.show()

    # formatter = TemporalFormatter(
    #     order_column=train_spec.problem.ordering,
    #     resample=True,
    #     resample_date_offset_unit='30S')
    # formatter.fit(dataframe)
    # # print(formatter.date_format)
    # result3 = formatter.transform(dataframe)
    # print(result3)


def test_format_dataframe_time_index_integer():
    dataframe = pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6]})
    _freq = format_dataframe_date_index(dataframe, order_column='a', date_offset_unit='years')
    assert dataframe.index.name == 'a'
    assert isinstance(dataframe.index, pd.DatetimeIndex)
    assert tuple(dataframe.index[:2]) == (datetime(1970, 1, 1), datetime(1971, 1, 1))


def test_format_dataframe_time_index_missing():
    dataframe = pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6]})
    _freq = format_dataframe_date_index(dataframe, order_column='d3mIndex')
    assert dataframe.index.name == 'd3mIndex'
    assert isinstance(dataframe.index, pd.DatetimeIndex)


def test_format_dataframe_time_index_format():
    dataframe = pd.DataFrame({'b': np.arange(220)})
    _freq = format_dataframe_date_index(dataframe, order_column='d3mIndex', date_offset_unit={'days': 1, 'months': 1})
    assert dataframe.index.name == 'd3mIndex'
    assert isinstance(dataframe.index, pd.DatetimeIndex)
    assert tuple(dataframe.index[:2]) == (datetime(1970, 1, 1), datetime(1970, 2, 2))


def test_problem_preprocessor():
    train_spec = TrainSpecification(problems['sunspots']['train_specification'])
    dataframe = Dataset(train_spec['input']).get_dataframe(usecols=train_spec.problem.train)

    formatter = ProblemPreprocessor(train_spec.problem)
    formatter.fit(dataframe)
    transformed = formatter.transform(dataframe)
    print("transformed", transformed)
    untransformed = formatter.inverse_transform(*transformed)
    print("untransformed", untransformed)


if __name__ == '__main__':
    # test_temporal_formatter_sunspots(plot=True)
    # test_format_dataframe_time_index_integer()
    # test_format_dataframe_time_index_missing()
    # test_format_dataframe_time_index_format()
    test_problem_preprocessor()

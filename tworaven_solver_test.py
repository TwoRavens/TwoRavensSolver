from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import tworaven_solver
import pandas
import pandas as pd
import numpy as np
from datetime import datetime

problems = {
    'shampoo': {
        'pipeline_specification': {
            'preprocess': None,
            'model': {
                'strategy': 'SARIMAX'
            }
        },
        'train_specification': {
            "problem": {
                "taskType": "FORECASTING",
                "predictors": [],
                "targets": ['Sales'],
                "time": ["Month"]
            },
            "input": {
                "name": "in-sample",
                "resource_uri": "file://" + '/ravens_volume/test_data/TR_TS_shampoo/TRAIN/dataset_TRAIN/tables/learningData.csv'
            }
        }
    },
    'appliance': {
        'pipeline_specification': {
            'preprocess': None,
            'model': {
                'strategy': 'VAR'
            }
        },
        'train_specification': {
            "problem": {
                "taskType": "FORECASTING",
                "predictors": ["T1", "T2"],
                "targets": ['Appliances'],
                "time": ["date"]
            },
            "input": {
                "name": "in-sample",
                "resource_uri": "file://" + '/ravens_volume/test_data/TR_TS_appliance/TRAIN/dataset_TRAIN/tables/learningData.csv'
            }
        }
    },
    'baseball': {
        'pipeline_specification': {
            'preprocess': "standard",
            'model': {
                'strategy': 'RANDOM_FOREST'
            }
        },
        'train_specification': {
            "problem": {
                "taskType": "CLASSIFICATION",
                "predictors": ["Games_played", "Number_seasons", 'Player'],
                "targets": ['Hall_of_Fame'],
                "categorical": ['Position', 'Player']
            },
            "input": {
                "name": "in-sample",
                "resource_uri": "file://" + "/ravens_volume/test_data/185_baseball/TRAIN/dataset_TRAIN/tables/learningData.csv"
            }
        }
    },
    'baseball-regression': {
        'pipeline_specification': {
            'preprocess': "standard",
            'model': {
                'strategy': 'ORDINARY_LEAST_SQUARES'
            }
        },
        'train_specification': {
            "problem": {
                "taskType": "REGRESSION",
                "predictors": ["Runs", "Hits", "At_bats"],
                "targets": ['Triples'],
                "categorical": ['Position', 'Player']
            },
            "input": {
                "name": "in-sample",
                "resource_uri": "file://" + "/ravens_volume/test_data/185_baseball/TRAIN/dataset_TRAIN/tables/learningData.csv"
            }
        }
    },
    'phem': {
        'pipeline_specification': {
            'preprocess': "standard",
            'model': {
                'strategy': 'ORDINARY_LEAST_SQUARES'
            }
        },
        'train_specification': {
            "problem": {
                "taskType": "REGRESSION",
                "predictors": ["Runs", "Hits", "At_bats"],
                "targets": ['Triples'],
                "categorical": ['Position', 'Player']
            },
            "input": {
                "name": "in-sample",
                "resource_uri": "file://" + "/ravens_volume/test_data/185_baseball/TRAIN/dataset_TRAIN/tables/learningData.csv"
            }
        }
    }
}

# Only support for time series forecasting

from tworaven_solver.search import SearchManager

problem = problems['appliance']
pip_spe, train_spe = problem['pipeline_specification'], problem['train_specification']
search_manager = SearchManager(None, train_spe['problem'])
new_pip = search_manager.get_pipeline_specification()

model = tworaven_solver.fit_pipeline(new_pip, train_spe)

dataframe = pd.read_csv(problem['train_specification']['input']['resource_uri'].replace('file://', ''))

# end = model.model.model._index[-1]
# start = model.model.model._index[0]
#
# model.model.plot_forecast()
#
# import matplotlib.pyplot as plt
#
# plt.show()
#
# dataframe = pandas.read_csv(data_path)
# dataframe['Month'] = pandas.to_datetime(dataframe['Month'])
# # dataframe = dataframe.set_index('Month')
#
#
# def infer_freq(series):
#
#     def approx_seconds(offset):
#         offset = pd.tseries.frequencies.to_offset(offset)
#         try:
#             return offset.nanos / 1E9
#         except ValueError:
#             pass
#
#         date = datetime.now()
#         return ((offset.rollback(date) - offset.rollforward(date)) * offset.n).total_seconds()
#
#     # infer frequency from every three-pair of records
#     candidate_frequencies = set()
#     for i in range(len(series) - 3):
#         candidate_frequency = pd.infer_freq(series[i:i + 3])
#         if candidate_frequency:
#             candidate_frequencies.add(candidate_frequency)
#
#     # sort inferred frequency by approximate time durations
#     return sorted([(i, approx_seconds(i)) for i in candidate_frequencies], key=lambda x: x[1])[-1][0]
#
#
# freq = infer_freq(dataframe['Month'])
#
# dataframe = dataframe.set_index('Month')
# dataframe_temp = dataframe.resample(freq).mean()
# numeric_columns = list(dataframe.select_dtypes(include=[np.number]).columns.values)
# categorical_columns = [i for i in dataframe.columns.values if i not in numeric_columns]
#
# for dropped_column in categorical_columns:
#     dataframe_temp[dropped_column] = dataframe[dropped_column]
#
# print(numeric_columns)
# print(categorical_columns)
# dataframe = pd.DataFrame(ColumnTransformer(transformers=[
#     ('numeric', SimpleImputer(strategy='median'), numeric_columns),
#     ('categorical', SimpleImputer(strategy='most_frequent'), categorical_columns)
# ]).fit_transform(dataframe_temp), index=dataframe_temp.index, columns=dataframe_temp.columns)
#
# print(dataframe)
#
#
# # split_index = int(len(dataframe) * .75)
# # history = dataframe.head(split_index)
# # future = dataframe.head(-split_index)
# #
# # print(model_tworavens.forecast(history, 3))
# # print(future.head(3))
# # train_specification['problem']['targets'].append('Sales')
# #
# # model_tworavens = tworaven_solver.fit_pipeline(
# #     pipeline_specification=pipeline_specification,
# #     train_specification=train_specification)
# #
# # print(model_tworavens.forecast(3))

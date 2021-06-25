problems = {
    'shampoo': {
        'pipeline_specification': {
            'preprocess': None,
            'model': {
                # 'strategy': 'SARIMAX'
                'strategy': 'AR_NN'
            }
        },
        'train_specification': {
            "problem": {
                'forecasting': True,
                "predictors": [],
                "targets": ['Sales'],
                "forecastingHorizon": {
                    "column": "Month",
                    "value": 10
                },
                "date_format": {
                    "Month": "%y-%m"
                },
                "performanceMetric": {"metric": "rootMeanSquaredError"},  # meanSquareError | meanAbsoluteError
            },
            "input": {
                "name": "in-sample",
                "resource_uri": "file://" + '/ravens_volume/test_data/TR_TS_shampoo/TRAIN/dataset_TRAIN/tables/learningData.csv'
            }
        }
    },
    'sunspots': {
        'pipeline_specification': {
            'preprocess': None,
            'model': {
                'strategy': 'AR_NN'
            }
        },
        'train_specification': {
            "problem": {
                "taskType": "REGRESSION",
                'forecasting': True,
                "predictors": [],
                "targets": ['sunspots'],
                "forecastingHorizon": {
                    "column": "year-month",
                    "value": 10
                },
                "date_format": {
                    "year-month": "%Y-%m"
                },
                "performanceMetric": {"metric": "meanSquaredError"}
            },
            "input": {
                "name": "in-sample",
                "resource_uri": "file://" + '/ravens_volume/test_data/56_sunspots_monthly/TRAIN/dataset_TRAIN/tables/learningData.csv'
            },
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
                "taskType": "REGRESSION",
                'forecasting': True,
                "predictors": ["T1", "T2"],
                "targets": ['Appliances'],
                "forecastingHorizon": {
                    "column": "date",
                    "value": 10
                },
                "performanceMetric": {"metric": "meanSquaredError"}
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
                "taskType": "REGRESSION",
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


strategies = {
    'FORECASTING': {
        'UNIVARIATE': [
            {
                'strategy': 'AR',
            },
            *[
                {
                    'strategy': 'SARIMAX',
                    'order': order,
                }
                for order in [(1, 0, 0), (1, 1, 1), (4, 1, 2), (2, 1, 0)]
            ]
        ],
        'MULTIVARIATE': [
            {
                'strategy': 'VAR'
            }
        ]
    },
    'CLASSIFICATION': {
        'BINARY': [
            {
                'strategy': 'LOGISTIC_REGRESSION'
            },
            {
                'strategy': 'SUPPORT_VECTOR_CLASSIFIER'
            }
        ],
        'MULTICLASS': [
            *[
                {
                    'strategy': 'RANDOM_FOREST',
                    'n_estimators': n_estimators
                } for n_estimators in [10, 100]
            ],
            {
                'strategy': 'SUPPORT_VECTOR_CLASSIFIER'
            }
        ],
        'MULTILABEL': [
            *[
                {
                    'strategy': 'RANDOM_FOREST',
                    'n_estimators': n_estimators
                } for n_estimators in [10, 100]
            ],
            {
                'strategy': 'SUPPORT_VECTOR_CLASSIFIER'
            }
        ]
    },
    'REGRESSION': {
        'UNIVARIATE': [
            {
                'strategy': 'ORDINARY_LEAST_SQUARES'
            }
        ]
    }
}


class SearchManager(object):
    def __init__(self, system_params, problem_specification):
        self.problem_specification = problem_specification
        self.system_params = system_params

        task = problem_specification['taskType']
        subtask = problem_specification.get('taskSubtype')

        # TODO: forecasting subtypes need rework
        if problem_specification['taskType'] == 'FORECASTING':
            variables = problem_specification['targets'] + problem_specification['predictors']
            subtask = 'MULTIVARIATE' if len(variables) > 2 else 'UNIVARIATE'

        if problem_specification['taskType'] == 'FORECASTING' and self.problem_specification.get('crossSection'):
            self.generator = iter(strategies.get(task, {}).get(subtask, []))
        else:
            self.generator = iter(strategies.get(task, {}).get(subtask, []))

    def get_pipeline_specification(self):
        try:
            model_specification = next(self.generator)
        except StopIteration:
            return

        return {
            'preprocess': None,
            'model': model_specification
        }

    def metalearn_result(self, pipeline_specification, scores):
        pass

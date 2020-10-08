
strategies = {
    'FORECASTING': {
        'UNIVARIATE': [
            {
                'strategy': 'AR',
                'library': 'statsmodels'
            },
            *[
                {
                    'strategy': 'SARIMAX_({},{},{})'.format(order[0], order[1], order[2]),
                    'library': 'statsmodels',
                    'order': order,
                }
                for order in [(1, 0, 0), (1, 1, 1), (4, 1, 2), (2, 1, 0), (0, 1, 2), (0, 1, 1), (0, 2, 2)]
            ],
            *[
                {
                    'strategy': 'AR_NN',
                    'library': 'sklearn',
                    'back_steps': step,
                } for step in [1, 2]
            ],
            {'strategy': 'TRA_AVERAGE', 'library': 'sklearn'},
            {'strategy': 'TRA_NAIVE', 'library': 'sklearn'},
            {'strategy': 'TRA_DRIFT', 'library': 'sklearn'},
        ],
        'MULTIVARIATE': [
            {
                'strategy': 'VAR',
                'library': 'statsmodels'
            },
            *[
                {
                    'strategy': 'VAR_NN',
                    'library': 'sklearn',
                    'back_steps': step,
                } for step in [1, 2, 3, 4]
            ],
            {'strategy': 'TRA_AVERAGE', 'library': 'sklearn'},
            {'strategy': 'TRA_NAIVE', 'library': 'sklearn'},
            {'strategy': 'TRA_DRIFT', 'library': 'sklearn'},
        ]
    },
    'CLASSIFICATION': {
        'BINARY': [
            {'strategy': 'LOGISTIC_REGRESSION', 'library': 'sklearn'},
            *[
                {
                    'strategy': 'RANDOM_FOREST',
                    'library': 'sklearn',
                    'n_estimators': n_estimators
                } for n_estimators in [10, 100]
            ],
            {'strategy': 'SUPPORT_VECTOR_CLASSIFIER', 'library': 'sklearn'},
            {"strategy": "RIDGE_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "RIDGE_CLASSIFIER_CV", 'library': 'sklearn'},
            {"strategy": "K_NEIGHBORS_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "DECISION_TREE_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "GRADIENT_BOOSTING_CLASSIFIER", "library": "sklearn"},
            {"strategy": "LINEAR_DISCRIMINANT_ANALYSIS", "library": "sklearn"},
            {"strategy": "QUADRATIC_DISCRIMINANT_ANALYSIS", "library": "sklearn"},
            # {"strategy": "GAUSSIAN_PROCESS_CLASSIFIER", "library": "sklearn"},
            {"strategy": "MULTINOMIAL_NAIVE_BAYES", "library": "sklearn"},
            {"strategy": "GAUSSIAN_NAIVE_BAYES", "library": "sklearn"},
            {"strategy": "COMPLEMENT_NAIVE_BAYES", "library": "sklearn"},
            {"strategy": "ADABOOST_CLASSIFIER", "library": "sklearn"},
            {'strategy': 'LOGISTIC_REGRESSION_CV', 'library': 'sklearn'},
        ],
        'MULTICLASS': [
            *[
                {
                    'strategy': 'RANDOM_FOREST',
                    'library': 'sklearn',
                    'n_estimators': n_estimators
                } for n_estimators in [10, 100]
            ],
            {'strategy': 'SUPPORT_VECTOR_CLASSIFIER', 'library': 'sklearn'},
            # {"strategy": "RIDGE_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "K_NEIGHBORS_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "DECISION_TREE_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "GRADIENT_BOOSTING_CLASSIFIER", "library": "sklearn"},
            {"strategy": "LINEAR_DISCRIMINANT_ANALYSIS", "library": "sklearn"},
            {"strategy": "QUADRATIC_DISCRIMINANT_ANALYSIS", "library": "sklearn"},
            # {"strategy": "GAUSSIAN_PROCESS_CLASSIFIER", "library": "sklearn"},
            {"strategy": "MULTINOMIAL_NAIVE_BAYES", "library": "sklearn"},
            {"strategy": "GAUSSIAN_NAIVE_BAYES", "library": "sklearn"},
            {"strategy": "COMPLEMENT_NAIVE_BAYES", "library": "sklearn"},
            {"strategy": "ADABOOST_CLASSIFIER", "library": "sklearn"},
        ],
        'MULTILABEL': [
            *[
                {
                    'strategy': 'RANDOM_FOREST',
                    'library': 'sklearn',
                    'n_estimators': n_estimators
                } for n_estimators in [10, 100]
            ],
            {'strategy': 'SUPPORT_VECTOR_CLASSIFIER', 'library': 'sklearn'},
            {"strategy": "RIDGE_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "K_NEIGHBORS_CLASSIFIER", 'library': 'sklearn'},
            {"strategy": "DECISION_TREE_CLASSIFIER", 'library': 'sklearn'},
        ]
    },
    'REGRESSION': {
        'UNIVARIATE': [
            {'strategy': 'ORDINARY_LEAST_SQUARES', 'library': 'sklearn'},
            {"strategy": "RANDOM_FOREST_REGRESSOR", 'library': 'sklearn'},
            {"strategy": "SUPPORT_VECTOR_REGRESSION", 'library': 'sklearn'},
            {"strategy": "K_NEIGHBORS_REGRESSOR", 'library': 'sklearn'},
            {"strategy": "DECISION_TREE_REGRESSOR", 'library': 'sklearn'},
            {"strategy": "LASSO_REGRESSION", "library": "sklearn"},
            {"strategy": "LASSO_REGRESSION_LARS", "library": "sklearn"},
            {"strategy": "ELASTIC_NET", "library": "sklearn"},
            {"strategy": "ORTHOGONAL_MATCHING", "library": "sklearn"},
            {"strategy": "ADABOOST_REGRESSOR", "library": "sklearn"},
            {"strategy": "GRADIENT_BOOSTING_REGRESSOR", "library": "sklearn"},
            # {"strategy": "GAUSSIAN_PROCESS_REGRESSOR", "library": "sklearn"},
            {"strategy": "RIDGE_CV", "library": "sklearn"},
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
            tmp_cross = problem_specification.get('crossSection', [])
            variables = [var for var in variables if var not in tmp_cross]
            subtask = 'MULTIVARIATE' if len(variables) > 2 else 'UNIVARIATE'

        # if problem_specification['taskType'] == 'FORECASTING' and self.problem_specification.get('crossSection'):
        self.generator = iter(strategies.get(task, {}).get(subtask, []))
        # else:
        #     self.generator = iter(strategies.get(task, {}).get(subtask, []))

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

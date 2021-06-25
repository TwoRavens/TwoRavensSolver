import copy

from tworaven_solver.utilities import ProblemSpecification

# TODO: all keywords should have both positive and negative forms, to keep issubset from admitting false-positives
_base_strategies_regression = [
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {'strategy': 'ORDINARY_LEAST_SQUARES', 'library': 'sklearn'}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "RANDOM_FOREST_REGRESSOR", 'library': 'sklearn'}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "SUPPORT_VECTOR_REGRESSION", 'library': 'sklearn'}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "K_NEIGHBORS_REGRESSOR", 'library': 'sklearn'}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "DECISION_TREE_REGRESSOR", 'library': 'sklearn'}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "LASSO_REGRESSION", "library": "sklearn"}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "LASSO_REGRESSION_LARS", "library": "sklearn"}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "ELASTIC_NET", "library": "sklearn"}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "ORTHOGONAL_MATCHING", "library": "sklearn"}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "ADABOOST_REGRESSOR", "library": "sklearn"}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "GRADIENT_BOOSTING_REGRESSOR", "library": "sklearn"}
    },
    {
        'keywords': ('REGRESSION', 'UNIVARIATE'),
        'model': {"strategy": "RIDGE_CV", "library": "sklearn"}
    },
    # {
    #     'keywords': ('REGRESSION', 'UNIVARIATE'),
    #     'model': {"strategy": "DUMMY_REGRESSOR", "library": "sklearn", "strategy": "mean"}
    # },
]
_base_strategies_classification = [
    {
        'keywords': ('CLASSIFICATION', 'BINARY', 'UNIVARIATE'),
        'model': {'strategy': 'LOGISTIC_REGRESSION', 'library': 'sklearn'}
    },
    *[{
        'keywords': ('CLASSIFICATION', 'BINARY', 'MULTICLASS', 'UNIVARIATE', 'MULTIVARIATE'),
        'model': {
            'strategy': 'RANDOM_FOREST',
            'library': 'sklearn',
            'n_estimators': n_estimators
        }
    } for n_estimators in [10, 100]],
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTILABEL'),
        "model": {"strategy": "SUPPORT_VECTOR_CLASSIFIER", "library": "sklearn", "probability": True}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTILABEL'),
        "model": {"strategy": "RIDGE_CLASSIFIER", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTILABEL'),
        "model": {"strategy": "RIDGE_CLASSIFIER_CV", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS', 'MULTILABEL'),
        "model": {"strategy": "K_NEIGHBORS_CLASSIFIER", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS', 'MULTILABEL'),
        "model": {"strategy": "DECISION_TREE_CLASSIFIER", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS'),
        "model": {"strategy": "GRADIENT_BOOSTING_CLASSIFIER", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS'),
        "model": {"strategy": "LINEAR_DISCRIMINANT_ANALYSIS", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS'),
        "model": {"strategy": "QUADRATIC_DISCRIMINANT_ANALYSIS", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS'),
        "model": {"strategy": "MULTINOMIAL_NAIVE_BAYES", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS'),
        "model": {"strategy": "GAUSSIAN_NAIVE_BAYES", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS'),
        "model": {"strategy": "COMPLEMENT_NAIVE_BAYES", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY", 'MULTICLASS'),
        "model": {"strategy": "ADABOOST_CLASSIFIER", "library": "sklearn"}
    },
    {
        "keywords": ("CLASSIFICATION", "BINARY"),
        "model": {"strategy": "LOGISTIC_REGRESSION_CV", "library": "sklearn"}
    },
    # {
    #     'keywords': ('REGRESSION', 'UNIVARIATE'),
    #     'model': {"strategy": "DUMMY_CLASSIFIER", "library": "sklearn", "strategy": "mean"}
    # },
]

_strategies_regression_prediction = copy.deepcopy(_base_strategies_regression)
for strategy in _strategies_regression_prediction:
    strategy['keywords'] = (*strategy['keywords'], 'PREDICTION')

_strategies_classification_prediction = copy.deepcopy(_base_strategies_classification)
for strategy in _strategies_classification_prediction:
    strategy['keywords'] = (*strategy['keywords'], 'PREDICTION')

_strategies_regression_forecasting = [
    # baselines
    *[{
        'keywords': ('REGRESSION', 'FORECASTING', 'UNIVARIATE', 'MULTIVARIATE'),
        'preprocess': {'resample': True},
        'model': {'strategy': 'BASELINE_REGRESSOR', 'library': 'sklearn', 'method': method}
    } for method in ('MEAN', 'NAIVE', 'DRIFT')],
    # sarimax
    *[{
        'keywords': ('REGRESSION', 'FORECASTING', 'UNIVARIATE'),
        'preprocess': {'resample': True},
        'model': {
            'strategy': 'SARIMAX',
            'library': 'statsmodels',
            'order': order,
        }
    } for order in [(4, 1, 2), (1, 1, 1), (4, 1, 2), (2, 1, 0), (0, 1, 2), (0, 1, 1), (0, 2, 2)]],
    # auto-regression
    # *[{
    #     'keywords': ('REGRESSION', 'FORECASTING', 'UNIVARIATE'),
    #     'preprocess': {'resample': True},
    #     'model': {
    #         'strategy': 'AR_NN',
    #         'library': 'sklearn',
    #         'back_steps': step,
    #     }
    # } for step in [1, 2]],
    # # neural net regression
    # *[{
    #     'keywords': ('REGRESSION', 'FORECASTING', 'MULTIVARIATE'),
    #     'preprocess': {'resample': True},
    #     'model': {
    #         'strategy': 'VAR_NN',
    #         'library': 'sklearn',
    #         'back_steps': step,
    #     }} for step in [1, 2, 3, 4]
    # ],
    # auto-regression
    {
        'keywords': ('REGRESSION', 'FORECASTING', 'MULTIVARIATE'),
        'preprocess': {'resample': True},
        'model': {'strategy': 'AR', 'library': 'statsmodels', 'lags': list(range(1, 4))}
    },
    # vector-auto-regression
    {
        'keywords': ('REGRESSION', 'FORECASTING', 'MULTIVARIATE'),
        'preprocess': {'resample': True},
        'model': {'strategy': 'VAR', 'library': 'statsmodels'}
    },
    # windowed predictive regression
    # *[{
    #     'keywords': (*strategy['keywords'], 'FORECASTING'),
    #     'preprocess': dict(window=True, **strategy.get('preprocess', {})),
    #     'model': strategy['model']
    # } for strategy in _base_strategies_regression]
]

_strategies_classification_forecasting = [
    {
        "keywords": ("CLASSIFICATION", "MULTICLASS", "FORECASTING"),
        "model": {"strategy": "HIDDEN_MARKOV_MODEL", "library": "hmmlearn"}
    },
    *[{
        'keywords': (*strategy['keywords'], 'FORECASTING'),
        'preprocess': dict(window=True, **strategy.get('preprocess', {})),
        'model': strategy['model']
    } for strategy in _base_strategies_classification]
]

strategies = [
    *_strategies_regression_prediction,
    *_strategies_classification_prediction,

    *_strategies_regression_forecasting,
    *_strategies_classification_forecasting,
]


class SearchManager(object):
    def __init__(self, problem_specification, system_params=None):
        self.system_params = system_params
        self.problem = ProblemSpecification(problem_specification)
        keywords = set(self.problem.keywords)

        keywords.add("FORECASTING" if self.problem.is_forecasting else "PREDICTION")

        if self.problem.is_forecasting and self.problem.cross_sections:
            self._strategies = [{
                'preprocess': False,
                'model': {'library': 'cross-sectional', 'strategy': strategy}
            } for strategy in strategies if keywords.issubset(strategy['keywords'])]
        else:
            self._strategies = [strategy for strategy in strategies if keywords.issubset(strategy['keywords'])]

    def __iter__(self):
        self.current_strategy = 0
        return self

    def __next__(self):
        if self.current_strategy < len(self._strategies):
            self.current_strategy += 1
            return self._strategies[self.current_strategy - 1]
        else:
            raise StopIteration

    def metalearn_result(self, pipeline_specification, scores):
        pass


def find_pipeline(strategy):
    return next((i for i in strategies if i.get('model', {}).get('strategy') == strategy), None)

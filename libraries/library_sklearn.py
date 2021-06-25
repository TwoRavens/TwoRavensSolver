import inspect
import json

from sklearn.base import RegressorMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import \
    RandomForestClassifier, RandomForestRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, \
    Lasso, LassoLars, ElasticNet, OrthogonalMatchingPursuit, RidgeClassifierCV, LogisticRegressionCV, \
    RidgeCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tworaven_solver.solution import Estimator
from tworaven_solver.utilities import filter_args

import numpy as np
import pandas as pd

# raise explicit errors that have to be handled; don't just throw warnings
pd.options.mode.chained_assignment = 'raise'


class BaselineRegressor(RegressorMixin):
    def __init__(self, method=None):
        """
        :param method: one of {'MEAN', 'NAIVE', 'DRIFT'}
        """
        assert method in ('MEAN', 'NAIVE', 'DRIFT')
        self.method = method
        self._value = None
        self._slope = None
        self._length = None

    def fit(self, X, y):
        self._length = len(y)
        if self.method == 'MEAN':
            self._value = np.nanmean(y, axis=0)
        if self.method == 'NAIVE':
            self._value = y.iloc[-1, 0]
        if self.method == 'DRIFT':
            self._value = y.iloc[-1, 0]
            self._slope = (y.iloc[-1, 0] - y.iloc[0, 0]) / self._length

    def fitted_values(self):
        fitted = np.repeat(self._value, self._length, axis=0)
        # np.atleast_2d for trailing axes
        fitted.shape += (1,) * (2 - fitted.ndim)

        if self.method == 'DRIFT':
            fitted += (np.arange(self._length) - self._length)[:, None] * self._slope[None]
        return fitted

    def predict(self, X):
        prediction = np.repeat(self._value, len(X), axis=0)
        # np.atleast_2d for trailing axes
        prediction.shape += (1,) * (2 - prediction.ndim)

        if self.method == 'DRIFT':
            # print(self._slope.shape)
            # print(prediction.shape)
            # print(((1 + np.arange(len(X)))[:, None] * self._slope[None]).shape)
            prediction += (1 + np.arange(len(X)))[:, None] * self._slope[None]
        return prediction


sklearn_classes = {
    'ORDINARY_LEAST_SQUARES': LinearRegression,
    'LOGISTIC_REGRESSION': LogisticRegression,
    'LOGISTIC_REGRESSION_CV': LogisticRegressionCV,
    'RANDOM_FOREST': RandomForestClassifier,
    'SUPPORT_VECTOR_CLASSIFIER': SVC,
    "RIDGE_CLASSIFIER": RidgeClassifier,
    "RIDGE_CLASSIFIER_CV": RidgeClassifierCV,
    "RIDGE_CV": RidgeCV,
    "RANDOM_FOREST_REGRESSOR": RandomForestRegressor,
    "SUPPORT_VECTOR_REGRESSION": SVR,
    "K_NEIGHBORS_CLASSIFIER": KNeighborsClassifier,
    "K_NEIGHBORS_REGRESSOR": KNeighborsRegressor,
    "DECISION_TREE_CLASSIFIER": DecisionTreeClassifier,
    "DECISION_TREE_REGRESSOR": DecisionTreeRegressor,
    "LASSO_REGRESSION": Lasso,
    "LASSO_REGRESSION_LARS": LassoLars,
    "ELASTIC_NET": ElasticNet,
    "ORTHOGONAL_MATCHING": OrthogonalMatchingPursuit,
    "ADABOOST_CLASSIFIER": AdaBoostClassifier,
    "ADABOOST_REGRESSOR": AdaBoostRegressor,
    "GRADIENT_BOOSTING_CLASSIFIER": GradientBoostingClassifier,
    "GRADIENT_BOOSTING_REGRESSOR": GradientBoostingRegressor,
    "LINEAR_DISCRIMINANT_ANALYSIS": LinearDiscriminantAnalysis,
    "QUADRATIC_DISCRIMINANT_ANALYSIS": QuadraticDiscriminantAnalysis,
    "GAUSSIAN_PROCESS_CLASSIFIER": GaussianProcessClassifier,
    "GAUSSIAN_PROCESS_REGRESSOR": GaussianProcessRegressor,
    "MULTINOMIAL_NAIVE_BAYES": MultinomialNB,
    "GAUSSIAN_NAIVE_BAYES": GaussianNB,
    "COMPLEMENT_NAIVE_BAYES": ComplementNB,
    "DUMMY_REGRESSOR": DummyRegressor,
    "DUMMY_CLASSIFIER": DummyClassifier,
    "BASELINE_REGRESSOR": BaselineRegressor,
}


class SciKitLearnEstimator(Estimator):
    library = 'sklearn'

    def describe(self):
        def is_serializable(v):
            try:
                json.dumps(v)
                return True
            except Exception:
                return False

        def get_params(model):
            if issubclass(type(model), dict):
                return {k: get_params(v) for k, v in model.items()}
            if hasattr(model, 'get_params'):
                return {
                    k: v for k, v in model.get_params().items()
                    if k not in ['warm_start', 'verbose', 'n_jobs'] and v is not None and is_serializable(v)
                }
        return {
            'all_parameters': get_params(self.estimator)
        }

    def fitted_values(self):
        return self.fitted

    def predict(self, data):
        # special-case for the forecasting problem
        if self.problem.is_forecasting:
            data['X'] = data['y']

        return pd.DataFrame(
            data=self.estimator.predict(data['X']),
            columns=self.problem.targets,
            index=data.get('index'))

    def predict_proba(self, data):
        if not hasattr(self.estimator, 'predict_proba') and hasattr(self.estimator, 'decision_function'):
            return self.estimator.decision_function(data['X'])
        return self.estimator.predict_proba(data['X'])

    def fit(self, data, data_specification=None):
        """
        Fit and return self.

        :param data:
        :param data_specification:
        """
        sklearn_class = sklearn_classes[self.model_specification['strategy']]
        self.estimator = sklearn_class(
            **filter_args(
                self.model_specification,
                list(inspect.signature(sklearn_class.__init__).parameters.keys())))

        self.refit(data=data, data_specification=data_specification)
        return self

    def refit(self, data=None, data_specification=None):
        if self.data_specification and json.dumps(self.data_specification) == json.dumps(data_specification):
            return

        # special-case for the forecasting problem
        if self.problem.is_forecasting:
            data['X'] = data['y']

        self.estimator.fit(**filter_args({
            'X': data['X'],
            'y': data['y'],
            'sample_weight': data.get('weight')
        }, list(inspect.signature(self.estimator.fit).parameters.keys())))

        if self.problem.is_forecasting:
            self.fitted = self.predict(data)

        if hasattr(self.estimator, 'classes_'):
            # This is a classification problem, store the label for future use
            tmp_list = self.estimator.classes_.tolist()
            self.problem['clf_classes'] = [str(item) for item in tmp_list]

        self.data_specification = data_specification
        return self

import abc
import json
import os
import warnings
from typing import Optional, Type

import joblib
import pandas as pd

from tworaven_solver.transformers.general import ProblemPreprocessor
from tworaven_solver.utilities import TrainSpecification


class Estimator(object):
    library = None

    def __init__(
            self,
            model_specification,
            train_specification: TrainSpecification,
            data_specification=None,
            estimator=None,
            solution_dir=None):
        self.model_specification = model_specification
        self.train_specification = TrainSpecification(train_specification)
        self.data_specification = data_specification

        if estimator is not None:
            self.estimator = estimator
        if solution_dir is not None:
            self.estimator = self._load_estimator(solution_dir)

    def describe(self):
        raise NotImplementedError(f"{self.library} does not implement describe")

    @property
    def problem(self):
        return self.train_specification.problem

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def predict(self, dataframe):
        """
        Predict the target variables.
        :param dataframe:
        """
        pass

    def predict_proba(self, dataframe):
        """
        Predict the probabilities of the categories of the target variables.
        :param dataframe:
        :return per-observation probabilities of each target category
        """
        raise NotImplementedError(f"{self.library} does not implement predict_proba")

    def fitted_values(self):
        """
        Return the in-sample predictions on the training data.
        :return: dataframe
        """
        raise NotImplementedError(f"{self.library} does not implement fitted_values")

    @abc.abstractmethod
    def refit(self, dataframe=None, data_specification=None):
        """
        Refit the model parameters without changing the hyperparameters.
        :param dataframe:
        :param data_specification:
        """
        pass

    def save(self, solution_dir):
        """
        Serialize the model to disk at the given path.
        :return:
        """
        os.makedirs(solution_dir, exist_ok=True)
        joblib.dump(self.estimator, os.path.join(solution_dir, 'estimator.joblib'))

    @staticmethod
    def load(solution_dir):
        """
        Load a raven_solver estimator from disk.
        :param solution_dir:
        :return: an instance of a subclass of Estimator
        """

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'r') as metadata_path:
            metadata = json.load(metadata_path)

        SubEstimator = Estimator.get_subclass(metadata['library'])
        return SubEstimator(
            model_specification=metadata['pipeline_specification']['model'],
            train_specification=metadata['train_specification'],
            solution_dir=solution_dir)

    @staticmethod
    def _load_estimator(solution_dir):
        """
        only load the estimator
        :param solution_dir:
        :return:
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return joblib.load(os.path.join(solution_dir, 'estimator.joblib'))

    @staticmethod
    def get_subclass(library: str):
        if library == 'cross-sectional':
            from tworaven_solver.libraries.library_cross_sectional import CrossSectionalEstimator as SubEstimator
        elif library == 'sklearn':
            from tworaven_solver.libraries.library_sklearn import SciKitLearnEstimator as SubEstimator
        elif library == 'prophet':
            from tworaven_solver.libraries.library_prophet import ProphetEstimator as SubEstimator
        elif library == 'statsmodels':
            from tworaven_solver.libraries.library_statsmodels import StatsModelsEstimator as SubEstimator
        else:
            raise ValueError(f'Unrecognized library: {library}')

        return SubEstimator


class Solution(object):
    def __init__(
            self, *,
            pipeline_specification,
            train_specification: TrainSpecification,
            data_specification=None,

            preprocessor: Optional[ProblemPreprocessor] = None,
            estimator: Optional[Estimator] = None):

        self.pipeline_specification = pipeline_specification
        self.train_specification = TrainSpecification(train_specification)
        self.data_specification = data_specification

        self.preprocessor: Optional[ProblemPreprocessor] = preprocessor
        self.estimator: Optional[Estimator] = estimator

        # performanceMetric is relevant in situations where only problem_specification is known
        self.problem['performanceMetric'] = self.train_specification.performance_metric

    def describe(self):
        return self.estimator.describe()

    @property
    def problem(self):
        return self.train_specification.problem

    def fit(self, data=None):
        """
        fit a model based on json descriptions of the expected model, problem, and data pointer
        """

        if data is None:
            data = self.train_specification.dataset('input').get_dataframe()

        # 1. fit preprocessor
        if self.pipeline_specification.get('preprocess') is not False:
            self.preprocessor = ProblemPreprocessor(self.problem, self.pipeline_specification.get('preprocess'))
            self.preprocessor.fit(data)
            data = self.preprocessor.transform(data)

        # 2. fit estimator
        print(self.train_specification.spec)
        model_spec = self.pipeline_specification['model']
        SubEstimator: Type[Estimator] = Estimator.get_subclass(model_spec['library'])
        self.estimator = SubEstimator(model_spec, self.train_specification)
        self.estimator.fit(data)

        return self

    def predict(self, data):
        """
        Predict the target variables.
        :param data:
        """
        if self.preprocessor:
            data = self.preprocessor.transform(data)

        data = self.estimator.predict(data)

        if self.preprocessor:
            data = self.preprocessor.inverse_transform(y=data)

        return data

    def predict_proba(self, data):
        """
        Predict the target variables.
        :param data:
        """
        if self.preprocessor:
            data = self.preprocessor.transform(data)

        return self.estimator.predict_proba(data)

    def fitted_values(self):
        data = self.estimator.fitted_values()

        if self.preprocessor:
            data = self.preprocessor.inverse_transform(y=data)

        return data

    def get_future_dataframe(self, steps):
        """
        build a dataframe of future observations to be used as input to predict
        solution must not have any exogenous variables
        :param steps:
        :return: pd.DataFrame
        """
        if not self.problem.is_forecasting:
            raise NotImplementedError('get_future_dataframe is only implemented for forecasting problems')

        if self.problem.exogenous:
            raise ValueError('cannot build future dataframe without exogenous variables')

        preprocessor = self.preprocessor.transformer_y['temporal']
        offset = preprocessor.resample_date_offset_unit or preprocessor.input_date_offset_unit
        last_obs = preprocessor.date_offset_end + offset

        return pd.DataFrame(index=pd.date_range(last_obs, periods=steps, freq=offset))

    def refit(self, data=None, data_specification=None):
        """
        Refit the model parameters without changing the hyperparameters.
        :param data:
        :param data_specification:
        """
        if self.preprocessor:
            self.preprocessor.fit(data)
            data = self.preprocessor.transform(data)
        self.estimator.fit(data)
        return self

    @abc.abstractmethod
    def save(self, solution_dir=None):
        """
        Serialize the model to disk at the given path.
        :return:
        """
        os.makedirs(solution_dir, exist_ok=True)
        if self.preprocessor:
            joblib.dump(self.preprocessor, os.path.join(solution_dir, f'preprocess.joblib'))
        if self.estimator:
            self.estimator.save(solution_dir)

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'w') as metadata_file:
            json.dump({
                'pipeline_specification': self.pipeline_specification,
                'train_specification': self.train_specification.spec,
                'data_specification': self.data_specification,
                'library': self.estimator.library
            }, metadata_file)

    @staticmethod
    def load(solution_dir, metadata=None):
        """
        Load a raven_solver solution from disk.
        :param solution_dir:
        :param metadata:
        :return: a restored solution
        """

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'r') as metadata_path:
            metadata = json.load(metadata_path)

        preprocessor_path = os.path.join(solution_dir, 'preprocess.joblib')
        preprocessor = None
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)

        return Solution(
            pipeline_specification=metadata['pipeline_specification'],
            train_specification=metadata['train_specification'],
            data_specification=metadata.get('data_specification'),

            preprocessor=preprocessor,
            estimator=Estimator.load(solution_dir=solution_dir))

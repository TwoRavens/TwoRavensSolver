import abc
import json
import os


class BaseModelWrapper(object):
    def __init__(
            self,
            pipeline_specification,
            problem_specification,
            model,
            preprocessors=None,
            data_specification=None):

        self.pipeline_specification = pipeline_specification
        self.problem_specification = problem_specification
        self.data_specification = data_specification

        self.model = model
        self.preprocessors = preprocessors

    @abc.abstractmethod
    def predict(self, dataframe):
        """
        Predict the target variables.
        :param dataframe:
        """
        pass

    @abc.abstractmethod
    def refit(self, dataframe=None, data_specification=None):
        """
        Refit the model parameters without changing the hyperparameters.
        :param data_specification:
        :param dataframe:
        """
        pass

    @abc.abstractmethod
    def save(self, solution_dir):
        """
        Serialize the model to disk at the given path.
        :return:
        """
        pass

    @staticmethod
    def load(solution_dir, metadata=None):
        """
        Load a raven_solver model from disk.
        :param metadata:
        :param solution_dir:
        :return: an instance of a subclass of type(self)
        """

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'r') as metadata_path:
            metadata = json.load(metadata_path)

        if metadata['library'] == 'statsmodels':
            from tworaven_solver.libraries.library_statsmodels import StatsModelsWrapper
            return StatsModelsWrapper.load(solution_dir, metadata=metadata)
        elif metadata['library'] == 'scikit-learn':
            from tworaven_solver.libraries.library_sklearn import SciKitLearnWrapper
            return SciKitLearnWrapper.load(solution_dir, metadata=metadata)
        elif metadata['library'] == 'prophet':
            from tworaven_solver.libraries.library_prophet import ProphetWrapper
            return ProphetWrapper.load(solution_dir, metadata=metadata)
        else:
            raise ValueError(f'Unrecognized library: {metadata["library"]}')

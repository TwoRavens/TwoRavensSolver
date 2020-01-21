import abc

import joblib

from .utilities import format_dataframe_time_index, Dataset
from tworaven_solver.fit import fit_model

from statsmodels.tsa.ar_model import ARResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

import json
import os
import pandas as pd


class BaseModelWrapper(object):
    def __init__(
            self,
            pipeline_specification,
            problem_specification,
            model,
            preprocess=None,
            data_specification=None):

        self.pipeline_specification = pipeline_specification
        self.problem_specification = problem_specification
        self.data_specification = data_specification

        self.model = model
        self.preprocess = preprocess

    @abc.abstractmethod
    def predict(self, dataframe):
        """
        Predict the target variables.
        @param dataframe:
        """
        pass

    @abc.abstractmethod
    def forecast(self, dataframe_history, horizon=1):
        """
        Forecasts the next observation, horizon steps ahead
        @param dataframe_history: historical data from which to forecast from
        @param horizon: number of time steps ahead to predict
        """
        pass

    @abc.abstractmethod
    def refit(self, dataframe=None, data_specification=None):
        """
        Refit the model parameters without changing the hyperparameters.
        @param data_specification:
        @param dataframe:
        """
        pass

    @abc.abstractmethod
    def save(self, solution_dir):
        """
        Serialize the model to disk at the given path.
        @return:
        """
        pass

    @staticmethod
    def load(solution_dir, metadata=None):
        """
        Load a raven_solver model from disk.
        Should also load in-sample data for forecasting problems
        @param metadata:
        @param solution_dir:
        @return: an instance of a subclass of type(self)
        """

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'r') as metadata_path:
            metadata = json.load(metadata_path)

        if metadata['library'] == 'statsmodels':
            return StatsModelsWrapper.load(solution_dir, metadata=metadata)
        else:
            raise ValueError(f'Unrecognized library: {metadata["library"]}')


class StatsModelsWrapper(BaseModelWrapper):
    library = 'statsmodels'

    def predict(self, dataframe):
        pass

    def forecast(self, start, end):
        time = next(iter(self.problem_specification.get('time', [])), None)
        if type(self.model) is ARResultsWrapper:
            # self.model.model.endog = dataframe_history
            # standardize to dataframe
            predictions = self.model.predict(
                start=max(start, self.model.model._index[self.model.model.k_ar]),
                end=end).to_frame(name=self.problem_specification['targets'][0])

            if self.model.model._index[self.model.model.k_ar] > start:
                predictions = pd.concat([
                    pd.DataFrame(index=pd.date_range(
                        start, self.model.model._index[self.model.model.k_ar],
                        freq=self.model.model._index_freq)),
                    predictions
                ])
            # index name is missing
            if time:
                predictions.index.name = time
            return predictions

        if type(self.model) is VARResultsWrapper:
            # self.model.model.endog = dataframe_history[self.problem_specification['targets']]
            # self.model.model.exog = dataframe_history[self.problem_specification['predictors']]

            # poor behavior from statsmodels needs manual cleanup- https://github.com/statsmodels/statsmodels/issues/3531#issuecomment-284108566
            # y parameter is a bug, deprecated in .11
            # predictions don't provide dates; dates reconstructed based on freq
            all = self.problem_specification['targets'] + self.problem_specification['predictors']
            exog = [i for i in self.problem_specification.get('exogenous') if i in all]
            endog = [i for i in all if i not in exog and i != time]

            predictions = pd.DataFrame(
                data=self.model.model.predict(self.model.model.params, start, end),
                columns=endog)

            # predictions[self.problem_specification['time']] = pd.date_range(
            #     start=self.model.dates[-1] + self.model.model._index_freq,
            #     freq=self.model.model._index_freq,
            #     periods=horizon)

            predictions[time] = pd.date_range(
                start=max(start, self.model.model._index[self.model.model.k_ar]),
                end=end,
                freq=self.model.model._index_freq)

            if self.model.model._index[self.model.model.k_ar] > start:
                predictions = pd.concat([
                    pd.DataFrame(index=pd.date_range(
                        start, self.model.model._index[self.model.model.k_ar],
                        freq=self.model.model._index_freq)),
                    predictions
                ])
            if time:
                predictions = predictions.set_index(time)
            return predictions

    def refit(self, dataframe=None, data_specification=None):
        if data_specification is not None and json.dumps(data_specification) == json.dumps(self.data_specification):
            return

        if dataframe is None:
            dataframe = Dataset(data_specification).get_dataframe()

        self.model = fit_model(
            dataframe=dataframe,
            model_specification=self.pipeline_specification['model'],
            problem_specification=self.problem_specification,
            start_params=self.model.params)

        self.data_specification = data_specification

    def save(self, solution_dir):
        # self.model.remove_data()
        os.makedirs(solution_dir, exist_ok=True)

        model_filename = 'model.pickle'
        preprocess_filename = 'preprocess.pickle'

        joblib.dump(self.model, os.path.join(solution_dir, model_filename))

        if self.preprocess:
            joblib.dump(self.preprocess, os.path.join(solution_dir, preprocess_filename))

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'w') as metadata_file:
            json.dump({
                'library': self.library,
                'pipeline_specification': self.pipeline_specification,
                'problem_specification': self.problem_specification,
                'data_specification': self.data_specification,
                'model_filename': model_filename,
                'preprocess_filename': preprocess_filename
            }, metadata_file)

    @staticmethod
    def load(solution_dir, metadata=None):

        if not metadata:
            metadata_path = os.path.join(solution_dir, 'solution.json')
            with open(metadata_path, 'r') as metadata_path:
                metadata = json.load(metadata_path)

        pipeline_specification = metadata['pipeline_specification']
        problem_specification = metadata['problem_specification']
        data_specification = metadata.get('data_specification')

        model_path = os.path.join(solution_dir, metadata['model_filename'])
        preprocess_path = os.path.join(solution_dir, metadata['preprocess_filename'])

        model = joblib.load(model_path)

        # reconstruct model with data it was trained on
        # if metadata['problem_specification']['taskType'] == 'FORECASTING' and data_specification:
        #     dataframe = Dataset(data_specification).get_dataframe()
        #
        #     exog_names = problem_specification['predictors']
        #     endog_names = problem_specification['targets']
        #     date_name = problem_specification.get('time')
        #
        #     dataframe = format_dataframe_time_index(dataframe, date_name)
        #     model.model.endog = dataframe[endog_names]
        #     if type(model) == VARResultsWrapper:
        #         model.model.exog = dataframe[exog_names]

        preprocess = None
        if os.path.exists(preprocess_path):
            preprocess = joblib.load(preprocess_path)

        return StatsModelsWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            data_specification=data_specification,
            model=model,
            preprocess=preprocess
        )

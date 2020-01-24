import abc

import joblib

from .utilities import format_dataframe_time_index, Dataset
from tworaven_solver.fit import fit_model

from statsmodels.tsa.ar_model import ARResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

import json
import os
import pandas as pd


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
        elif metadata['library'] == 'scikit-learn':
            return SciKitLearnWrapper.load(solution_dir, metadata=metadata)
        else:
            raise ValueError(f'Unrecognized library: {metadata["library"]}')


class SciKitLearnWrapper(BaseModelWrapper):
    library = 'scikit-learn'

    def describe(self):
        pass

    def predict(self, dataframe):
        dataframe = dataframe[self.problem_specification['predictors']]
        if self.preprocessors:
            dataframe = pd.DataFrame(
                data=self.preprocessors['predictors'].transform(dataframe),
                index=dataframe.index)

        return pd.DataFrame(
            data=self.model.predict(dataframe),
            columns=self.problem_specification['targets'],
            index=dataframe.index)

    def predict_proba(self, dataframe):
        dataframe = dataframe[self.problem_specification['predictors']]
        if self.preprocessors:
            dataframe = pd.DataFrame(
                data=self.preprocessors['predictors'].transform(dataframe),
                index=dataframe.index)
        return self.model.predict_proba(dataframe)

    def refit(self, dataframe=None, data_specification=None):
        if self.data_specification and json.dumps(self.data_specification) == json.dumps(data_specification):
            return

        targets = self.problem_specification['targets']
        predictors = self.problem_specification['predictors']
        weight = self.problem_specification['weights']

        if weight and weight[0] in predictors:
            predictors.remove(weight[0])

        data_predictors = dataframe[predictors]
        if self.preprocessors:
            data_predictors = pd.DataFrame(
                data=self.preprocessors['predictors'].transform(data_predictors),
                index=dataframe.index)

        self.model.fit(
            X=data_predictors,
            y=dataframe[targets[0]],
            sample_weight=dataframe[weight[0]] if weight else None)

        self.data_specification = data_specification

    def forecast(self, dataframe_history, horizon=1):
        pass

    def save(self, solution_dir):
        os.makedirs(solution_dir, exist_ok=True)

        model_filename = 'model.joblib'
        preprocess_folder = 'preprocess'

        joblib.dump(self.model, os.path.join(solution_dir, model_filename))

        if self.preprocessors:
            preprocess_dir = os.path.join(solution_dir, preprocess_folder)
            os.makedirs(preprocess_dir, exist_ok=True)
            for name in self.preprocessors:
                joblib.dump(self.preprocessors[name], os.path.join(preprocess_dir, f'{name}.joblib'))

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'w') as metadata_file:
            json.dump({
                'library': self.library,
                'pipeline_specification': self.pipeline_specification,
                'problem_specification': self.problem_specification,
                'data_specification': self.data_specification,
                'model_filename': model_filename,
                'preprocess_folder': preprocess_folder
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
        preprocess_dir = os.path.join(solution_dir, metadata['preprocess_folder'])

        model = joblib.load(model_path)

        preprocessors = {}
        if os.path.exists(preprocess_dir):
            for filename in os.listdir(preprocess_dir):
                preprocessors[filename.replace('.joblib', '')] = joblib.load(os.path.join(preprocess_dir, filename))

        return SciKitLearnWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            data_specification=data_specification,
            model=model,
            preprocessors=preprocessors
        )


class StatsModelsWrapper(BaseModelWrapper):
    library = 'statsmodels'

    def predict(self, dataframe):
        pass

    def describe(self):
        if type(self.model) is ARResultsWrapper:
            return {
                'model': f'AR({self.model.k_ar})',
                'description': 'Autoregressive model'
            }

        if type(self.model) is VARResultsWrapper:
            return {
                'model': f'VAR({self.model.k_ar})',
                'description': 'Vector Autoregressive model'
            }

        if type(self.model) is SARIMAXResultsWrapper:
            return {
                'model': f'SARIMAX({self.model.model.k_ar}, {self.model.model.k_diff}, {self.model.model.k_ma})',
                'description': f'Seasonal autoregressive integrated moving average with exogenous regressors. The three values indicate the number of AR parameters, number of differences, and number of MA parameters.'
            }

    def forecast(self, start, end):
        time = next(iter(self.problem_specification.get('time', [])), None)
        if type(self.model) is ARResultsWrapper:
            # self.model.model.endog = dataframe_history
            # standardize to dataframe
            predictions = self.model.predict(
                # don't predict before autoregressive terms are populated
                start=max(start, self.model.model._index[self.model.model.k_ar]),
                # include the end date by offsetting end by one
                end=end)\
                .to_frame(name=self.problem_specification['targets'][0])

            if self.model.model._index[self.model.model.k_ar] > start:
                predictions = pd.concat([
                    pd.DataFrame(index=pd.date_range(
                        start, self.model.model._index[self.model.model.k_ar - 1],
                        freq=self.model.model._index_freq)),
                    predictions
                ])
            # index name is missing
            if time:
                predictions.index.name = time
            return predictions

        if type(self.model) is VARResultsWrapper:
            endog_target_names = [i for i in self.problem_specification['targets'] if i != time]

            start_model = min(max(start, self.model.model._index[self.model.k_ar]), self.model.model._index[-1])
            end_model = max(start, end)

            predictions = pd.DataFrame(
                data=self.model.model.predict(
                    params=self.model.params,
                    start=start_model, end=end_model)[:, :len(endog_target_names)],
                columns=endog_target_names)

            # predictions don't provide dates; dates reconstructed based on freq
            predictions[time] = pd.date_range(
                start=start_model,
                end=end_model,
                freq=self.model.model._index_freq)

            if start_model > start:
                predictions = pd.concat([
                    pd.DataFrame(index=pd.date_range(
                        start, self.model.model._index[self.model.k_ar - 1],
                        freq=self.model.model._index_freq)),
                    predictions
                ])
            if start_model < start:
                predictions = predictions[predictions[time] >= start]

            if time:
                predictions = predictions.set_index(time)
            return predictions

        if type(self.model) is SARIMAXResultsWrapper:
            all = self.problem_specification['targets'] + self.problem_specification['predictors']
            exog_names = [i for i in self.problem_specification.get('exogenous', []) if i in all]
            endog = [i for i in all if i not in exog_names and i != time]

            predictions = pd.DataFrame(
                data=self.model.predict(start, end),
                columns=endog)
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

        model_filename = 'model.joblib'
        preprocess_folder = 'preprocessors'

        joblib.dump(self.model, os.path.join(solution_dir, model_filename))

        if self.preprocessors:
            preprocess_dir = os.path.join(solution_dir, preprocess_folder)
            os.makedirs(preprocess_dir, exist_ok=True)
            for name in self.preprocessors:
                joblib.dump(self.preprocessors[name], os.path.join(preprocess_dir, f'{name}.joblib'))

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'w') as metadata_file:
            json.dump({
                'library': self.library,
                'pipeline_specification': self.pipeline_specification,
                'problem_specification': self.problem_specification,
                'data_specification': self.data_specification,
                'model_filename': model_filename,
                'preprocess_folder': preprocess_folder
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
        preprocess_dir = os.path.join(solution_dir, metadata['preprocess_folder'])

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

        preprocessors = {}
        if os.path.exists(preprocess_dir):
            for filename in os.listdir(preprocess_dir):
                preprocessors[filename.replace('.joblib', '')] = joblib.load(os.path.join(preprocess_dir, filename))

        return StatsModelsWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            data_specification=data_specification,
            model=model,
            preprocessors=preprocessors
        )

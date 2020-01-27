import abc

import joblib

from .utilities import split_time_series, format_dataframe_time_index

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
        index = dataframe.index
        dataframe = dataframe[self.problem_specification['predictors']]
        if self.preprocessors:
            dataframe = pd.DataFrame(
                data=self.preprocessors['predictors'].transform(dataframe),
                index=dataframe.index)

        return pd.DataFrame(
            data=self.model.predict(dataframe),
            columns=self.problem_specification['targets'],
            index=index)

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

    def predict(self, dataframe):
        cross_section_names = self.problem_specification.get('crossSection', [])
        treatments_data = split_time_series(dataframe=dataframe, cross_section_names=cross_section_names)
        time_name = next(iter(self.problem_specification.get('time', [])), None)
        index_names = self.problem_specification.get('indexes', ['d3mIndex'])
        predictions = []
        predict = None

        print(self.model)
        for treatment_name in treatments_data:
            treatment = treatments_data[treatment_name]

            treatment = format_dataframe_time_index(
                treatment,
                date=time_name,
                granularity_specification=self.problem_specification.get('timeGranularity'))
            treatment.reset_index(inplace=True)

            start = treatment[time_name].iloc[0]
            end = treatment[time_name].iloc[-1]

            index = treatment[index_names]
            treatment.drop(index_names, inplace=True, axis=1)

            model = self.model[treatment_name]

            if self.pipeline_specification['model']['strategy'] == 'AR':
                # model.model.endog = dataframe_history
                # standardize to dataframe
                predict = model.predict(
                    # don't predict before autoregressive terms are populated
                    start=max(start, model.model._index[model.model.k_ar]),
                    # include the end date by offsetting end by one
                    end=end)\
                    .to_frame(name=self.problem_specification['targets'][0])

                if model.model._index[model.model.k_ar] > start:
                    predict = pd.concat([
                        pd.DataFrame(index=pd.date_range(
                            start, model.model._index[model.model.k_ar - 1],
                            freq=model.model._index_freq)),
                        predict
                    ])
                # index name is missing
                predict.index.name = time_name
                predict.reset_index()

            if self.pipeline_specification['model']['strategy'] == 'VAR':
                endog_target_names = [i for i in self.problem_specification['targets'] if i != time_name]

                start_model = min(max(start, model.model._index[model.k_ar]), model.model._index[-1])
                end_model = max(start, end)

                predict = pd.DataFrame(
                    data=model.model.predict(
                        params=model.params,
                        start=start_model, end=end_model)[:, :len(endog_target_names)],
                    columns=endog_target_names)

                # predictions don't provide dates; dates reconstructed based on freq
                predict[time_name] = pd.date_range(
                    start=start_model,
                    end=end_model,
                    freq=model.model._index_freq)

                if start_model > start:
                    predict = pd.concat([
                        pd.DataFrame(index=pd.date_range(
                            start, model.model._index[model.k_ar - 1],
                            freq=model.model._index_freq)),
                        predict
                    ])
                if start_model < start:
                    predict = predict[predict[time_name] >= start]

            if self.pipeline_specification['model']['strategy'] == 'SARIMAX':
                all = self.problem_specification['targets'] + self.problem_specification['predictors']
                exog_names = [i for i in self.problem_specification.get('exogenous', []) if i in all]
                endog = [i for i in all if i not in exog_names and i != time_name]

                predict = pd.DataFrame(
                    data=model.predict(start, end),
                    columns=endog)

            if predict:
                predict[index_names] = index
                for i, section in enumerate(cross_section_names):
                    predict[section] = treatment_name[i]
                predictions.append(predict)

        return pd.concat(predictions)

    def refit(self, dataframe=None, data_specification=None):
        pass
    #     if data_specification is not None and json.dumps(data_specification) == json.dumps(self.data_specification):
    #         return
    #
    #     if dataframe is None:
    #         dataframe = Dataset(data_specification).get_dataframe()
    #
    #     self.model = fit_model(
    #         dataframe=dataframe,
    #         model_specification=self.pipeline_specification['model'],
    #         problem_specification=self.problem_specification,
    #         start_params=self.model.params)
    #
    #     self.data_specification = data_specification

    def save(self, solution_dir):
        # self.model.remove_data()
        os.makedirs(solution_dir, exist_ok=True)

        model_folder = 'models'
        preprocess_folder = 'preprocessors'

        preprocess_dir = os.path.join(solution_dir, preprocess_folder)
        for treatment_name in self.preprocessors:
            for preprocessor_name in self.preprocessors[treatment_name]:
                preprocess_treatment_dir = os.path.join(preprocess_dir, str(hash(treatment_name)))
                os.makedirs(preprocess_treatment_dir, exist_ok=True)
                joblib.dump(
                    self.preprocessors[treatment_name][preprocessor_name],
                    os.path.join(preprocess_treatment_dir, f'{preprocessor_name}.joblib'))

        model_dir = os.path.join(solution_dir, model_folder)
        os.makedirs(model_dir, exist_ok=True)
        for treatment_name in self.model:
            joblib.dump(
                self.model[treatment_name],
                os.path.join(solution_dir, model_folder, f'{hash(treatment_name)}.json'))

        with open(os.path.join(solution_dir, 'model_treatments.csv'), 'w') as treatment_file:
            json.dump(
                [{'name': list(treatment_name) if type(treatment_name) is tuple else [treatment_name], 'id': hash(treatment_name)} for treatment_name in self.model],
                treatment_file)

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'w') as metadata_file:
            json.dump({
                'library': self.library,
                'pipeline_specification': self.pipeline_specification,
                'problem_specification': self.problem_specification,
                'data_specification': self.data_specification,
                'model_folder': model_folder,
                'preprocess_folder': preprocess_folder
            }, metadata_file)

    @staticmethod
    def load(solution_dir, metadata=None):

        print('solution dir', solution_dir)

        if not metadata:
            metadata_path = os.path.join(solution_dir, 'solution.json')
            with open(metadata_path, 'r') as metadata_path:
                metadata = json.load(metadata_path)

        pipeline_specification = metadata['pipeline_specification']
        problem_specification = metadata['problem_specification']
        data_specification = metadata.get('data_specification')

        model_dir = os.path.join(solution_dir, metadata['model_folder'])
        preprocess_dir = os.path.join(solution_dir, metadata['preprocess_folder'])

        with open(os.path.join(solution_dir, 'model_treatments.csv'), 'r') as treatment_file:
            treatments = json.load(treatment_file)

        preprocessors = {}
        models = {}
        for treatment in treatments:
            treatment['name'] = tuple(treatment['name'])
            model_path = os.path.join(model_dir, f'{treatment["id"]}.joblib')
            models[treatment['name']] = joblib.load(model_path)

            preprocess_path = os.path.join(preprocess_dir, treatment['name'])
            preprocessors[treatment['name']] = {}
            for preprocessor_name in os.listdir(preprocess_path):
                preprocessor_path = os.path.join(preprocess_path, f'{preprocessor_name}.joblib')
                preprocessors[treatment['name']][preprocessor_name] = joblib.load(preprocessor_path)

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

        return StatsModelsWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            data_specification=data_specification,
            model=models,
            preprocessors=preprocessors
        )

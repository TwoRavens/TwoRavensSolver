import abc
import inspect

import warnings
import joblib
# import torch

from .utilities import split_time_series, format_dataframe_time_index, filter_args, format_dataframe_order_index
from .fit import fit_forecast_preprocess

from statsmodels.tsa.ar_model import ARResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

import json
import os
import pandas as pd
import numpy as np


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

        # Categorical variable should be converted to string
        nominal = [i for i in self.problem_specification.get('categorical', [])
                   if i in self.problem_specification['predictors']]
        dataframe[nominal] = dataframe[nominal].astype(str)

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

        if 'predict_proba' in dir(self.model):
            return self.model.predict_proba(dataframe)
        else:
            return self.model.decision_function(dataframe)

    def forecast(self, dataframe, forecast_len, forecast_mode='test'):
        cross_section_names = self.problem_specification.get('crossSection', [])
        time_name = next(iter(self.problem_specification.get('time', [])), None)
        index_names = self.problem_specification.get('indexes', ['d3mIndex'])
        target_names = self.problem_specification['targets']

        treatments_data = split_time_series(dataframe=dataframe, cross_section_names=cross_section_names)
        predictions = []
        predict = None

        for treatment_name in treatments_data:
            treatment = treatments_data[treatment_name]
            # treatment = treatment.head(forecast_len)
            if type(treatment_name) is not tuple:
                pass
            if treatment_name not in self.model:
                print('unknown treatment:', treatment_name)
                continue
            model = self.model[treatment_name]

            treatment, mapping_dic = format_dataframe_order_index(
                treatment,
                is_date=self.problem_specification['is_temporal'],
                date_format=self.problem_specification['date_format'],
                order_column=time_name,
                freq=model._index.freq
            )
            treatment = treatment.head(forecast_len)

            index, date_index, original_time_index = treatment[index_names], treatment.index, None
            # treatment.drop(index_names, inplace=True, axis=1)
            # Deal with a very specific case that d3mIndex plays the role of time
            treatment.drop([eachIndex for eachIndex in index_names if eachIndex in treatment.columns],
                           inplace=True, axis=1)

            if mapping_dic and time_name in treatment.columns:
                # Dummy dateTimeIndex is activated, the original order-column should be dropped.
                treatment.drop(time_name, inplace=True, axis=1)
            index.reset_index(drop=True, inplace=True)
            dataframe_len, history_len = len(treatment.index), len(model._index)

            if self.pipeline_specification['model']['strategy'].endswith('_NN'):
                # May contains some issues -- corner cases
                if 'test' == forecast_mode:
                    # TEST SPLIT
                    predict = pd.DataFrame(
                        data=model.forecast(treatment, len(treatment.index), real_value=False),
                        columns=self.problem_specification['targets'],
                        index=date_index)
                elif 'train' == forecast_mode:
                    predict = pd.DataFrame(
                        data=model.forecast(treatment, len(treatment.index), real_value=True),
                        columns=self.problem_specification['targets'],
                        index=date_index)
                else:
                    train_part = model.forecast(treatment.head(history_len), history_len, real_value=True)
                    test_part = model.forecast(treatment.tail(dataframe_len - history_len),
                                               dataframe_len - history_len,
                                               real_value=False)
                    predict = pd.DataFrame(
                        data=np.concatenate((train_part, test_part), axis=0),
                        columns=self.problem_specification['targets'],
                        index=date_index
                    )

            if self.pipeline_specification['model']['strategy'].startswith('TRA_'):
                length = len(treatment.index)
                if model.method == 'AVERAGE' or model.method == 'NAIVE':
                    if len(model.value.shape) == 1:
                        model.value = model.value.reshape(1, -1)
                    data = np.repeat(model.value, length, axis=0)
                elif model.method == 'DRIFT':
                    slop, tmp = model.value['slope'], model.value['pivot']

                    if len(tmp.shape) == 1:
                        tmp = tmp.reshape(1, -1)
                    if len(slop.shape) != 1:
                        slop = slop.ravel()

                    data = np.repeat(tmp, length, axis=0)
                    # Drift method has different behavior in train/test split
                    if 'test' == forecast_mode:
                        # TEST SPLIT
                        for factor in range(length):
                            data[factor] += slop * (1 + factor)
                    elif 'train' == forecast_mode:
                        for factor in range(length - 1, -1, -1):
                            data[factor] -= slop * (length - factor - 1)
                    else:
                        for factor in range(history_len - 1, -1, -1):
                            data[factor] -= slop * (history_len - factor - 1)
                        # data[history_len] should be the pivot itself
                        for factor in range(history_len, dataframe_len):
                            data[factor] += slop * (factor - history_len + 1)

                else:
                    raise NotImplementedError('Unsupported method {} detected'.format(model['method']))

                predict = pd.DataFrame(
                    data=data,
                    columns=self.problem_specification['targets'],
                    index=date_index)

            if predict is not None:
                # removing data from below the asked-for interval, for example, can make the index start from non-zero
                predict.reset_index(drop=True, inplace=True)

                predict[index_names] = index

                if not isinstance(treatment_name, tuple):
                    treatment_name = (treatment_name,)

                for i, section in enumerate(cross_section_names):
                    predict[section] = treatment_name[i]

                predictions.append(predict)

        if not predictions:
            return pd.DataFrame(data=[], columns=[index_names[0], *target_names])

        # Copied post-process code from statsmodel
        predictions = pd.concat(predictions)

        # remove interpolated entries in the time series that cannot be matched back to the input dataframe
        predictions.dropna(inplace=True)
        # predictions = predictions[[*index_names, *target_names, time_name, *cross_section_names]]

        # cast d3mIndex back to int (it became float due to na values)
        predictions[index_names[0]] = predictions[index_names[0]].astype(int)

        # indices are duplicated after concat
        predictions.reset_index(drop=True, inplace=True)

        return predictions

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

        self.model.fit(**filter_args({
            'X': data_predictors,
            'y': dataframe[targets[0]],
            'sample_weight': dataframe[weight[0]] if weight else None
        }, list(inspect.signature(self.model.fit).parameters.keys())))

        self.data_specification = data_specification


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
        time_name = next(iter(self.problem_specification.get('time', [])), None)
        index_names = self.problem_specification.get('indexes', ['d3mIndex'])
        target_names = self.problem_specification['targets']
        # Default length is 10
        forecast_length = self.problem_specification.get('forecastingHorizon', {"value": 10})
        forecast_length = forecast_length.get('value', 10)

        treatments_data = split_time_series(dataframe=dataframe, cross_section_names=cross_section_names)
        predictions = []
        predict = None

        # print(list(treatments_data.keys())[:10])
        # print(list(self.model.keys())[:10])
        for treatment_name in treatments_data:
            treatment = treatments_data[treatment_name]
            if type(treatment_name) is not tuple:
                pass
            if treatment_name not in self.model:
                print('unknown treatment:', treatment_name)
                ava = [name for name in self.model]
                print('Available models:', ava)
                continue
            model = self.model[treatment_name]

            treatment, _ = format_dataframe_order_index(
                treatment,
                order_column=time_name,
                is_date=self.problem_specification['is_temporal'],
                date_format=self.problem_specification['date_format'],
                freq=model.model._index.freq
            )

            treatment.reset_index(inplace=True)

            start = treatment[time_name].iloc[0]
            end = treatment[time_name].iloc[-1]
            # end = treatment[time_name].iloc[-1] if len(treatment) > forecast_length \
            #     else treatment[time_name].iloc[forecast_length-1]

            index = treatment[index_names]
            treatment.drop(index_names, inplace=True, axis=1)
            # print('index', index)

            # Removed strategy ?
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
                            freq=model.model._index.freq)),
                        predict
                    ])
                # index name is missing
                predict.index.name = time_name
                predict.reset_index()

            if self.pipeline_specification['model']['strategy'] == 'VAR':
                endog_target_names = [i for i in self.problem_specification['targets'] if i != time_name]

                start_model = min(max(start, model.model._index[model.k_ar]), model.model._index[-1])
                end_model = max(start_model, end)

                new_time_index = pd.date_range(
                    start=start_model,
                    end=end_model,
                    freq=model.model._index.freq)

                predict = model.model.predict(
                    params=model.params,
                    start=start_model, end=end_model)

                if len(predict) == 0:
                    predict = np.empty(shape=(0, len(endog_target_names)))
                if len(predict) > len(new_time_index):
                    predict = predict[:len(new_time_index)]

                # predictions don't provide dates; dates reconstructed based on freq
                predict = pd.DataFrame(
                    data=predict[:, :len(endog_target_names)],
                    columns=endog_target_names)

                # print('date range: ', start, end)
                # print('model range: ', start_model, end_model)
                predict[time_name] = new_time_index

                if start_model > start:
                    # print('prepending data from before range')
                    predict = pd.concat([
                        pd.DataFrame(index=pd.date_range(
                            start, model.model._index[model.k_ar - 1],
                            freq=model.model._index.freq)),
                        predict
                    ])
                if start_model < start:
                    # print(predict)
                    # print('removing data from below the asked-for-interval, but was forecasted')
                    predict = predict[predict[time_name] >= start]
                    # print(predict)

            if self.pipeline_specification['model']['strategy'].startswith('SARIMAX'):
                all = self.problem_specification['targets'] + self.problem_specification['predictors']
                exog_names = [i for i in self.problem_specification.get('exogenous', []) if i in all]
                endog = [i for i in all if i not in exog_names and i != time_name]

                predict = pd.DataFrame(
                    data=model.predict(start, end),
                    columns=endog)
            # print(predict)
            if predict is not None:
                # removing data from below the asked-for interval, for example, can make the index start from non-zero
                predict.reset_index(drop=True, inplace=True)

                predict[index_names] = index

                if not isinstance(treatment_name, list):
                    treatment_name = [treatment_name]
                for i, section in enumerate(cross_section_names):
                    predict[section] = treatment_name[i]
                predictions.append(predict)

        if not predictions:
            return pd.DataFrame(data=[], columns=[index_names[0], *target_names])

        predictions = pd.concat(predictions)

        # remove interpolated entries in the time series that cannot be matched back to the input dataframe
        predictions.dropna(inplace=True)
        # predictions = predictions[[*index_names, *target_names, time_name, *cross_section_names]]

        # cast d3mIndex back to int (it became float due to na values)
        predictions[index_names[0]] = predictions[index_names[0]].astype(int)

        # indices are duplicated after concat
        predictions.reset_index(drop=True, inplace=True)
        return predictions

    def forecast(self, dataframe, forecast_len=None, forecast_mode='test'):
        cross_section_names = self.problem_specification.get('crossSection', [])
        time_name = next(iter(self.problem_specification.get('time', [])), None)
        index_names = self.problem_specification.get('indexes', ['d3mIndex'])
        target_names = self.problem_specification['targets']

        treatments_data = split_time_series(dataframe=dataframe, cross_section_names=cross_section_names)
        predictions = []
        predict = None

        for treatment_name in treatments_data:
            treatment = treatments_data[treatment_name]

            if type(treatment_name) is not tuple:
                treatment_name = (treatment_name,)
            if treatment_name not in self.model:
                print('unknown treatment:', treatment_name)
                continue
            model = self.model[treatment_name]

            treatment, _ = format_dataframe_order_index(
                treatment,
                order_column=time_name,
                is_date=self.problem_specification['is_temporal'],
                date_format=self.problem_specification['date_format'],
                freq=model.model._index.freq
            )

            treatment.reset_index(inplace=True)
            treatment = treatment.head(forecast_len)

            index = treatment[index_names]
            # Deal with a very specific case that d3mIndex plays the role of time
            treatment.drop([eachIndex for eachIndex in index_names if eachIndex in treatment.columns],
                           inplace=True, axis=1)

            # Removed strategy ?
            if self.pipeline_specification['model']['strategy'] == 'AR':
                if 'test' == forecast_mode:
                    predict = model.predict(
                        start=len(model.model._index) - 1,
                        end=len(model.model._index) + min(forecast_len, len(treatment.index) - 1)
                    ).to_frame(name=self.problem_specification['targets'][0])
                elif 'train' == forecast_mode:
                    predict = model.predict(
                        start=0,
                        end=min(forecast_len, len(treatment.index) - 1)
                    ).to_frame(name=self.problem_specification['targets'][0])
                else:
                    # All split
                    predict = model.predict(
                        start=0,
                        end=len(treatment.index) - 1
                    ).to_frame(name=self.problem_specification['targets'][0])

                predict.index.name = time_name
                predict.reset_index()

            if self.pipeline_specification['model']['strategy'] == 'VAR':
                endog_target_names = [i for i in self.problem_specification['targets'] if i != time_name]

                if 'test' == forecast_mode:
                    predict = model.model.predict(
                        params=model.params,
                        start=len(model.model._index) - 1,
                        end=len(model.model._index) + min(forecast_len, len(treatment.index) - 1)
                    )
                elif 'train' == forecast_mode:
                    predict = model.model.predict(
                        params=model.params,
                        start=0,
                        end=min(forecast_len, len(treatment.index) - 1)
                    )
                else:
                    # All split
                    predict = model.model.predict(
                        params=model.params,
                        start=0,
                        end=len(treatment.index) - 1
                    )

                if len(predict) == 0:
                    predict = np.empty(shape=(0, len(endog_target_names)))

                # predictions don't provide dates; dates reconstructed based on freq
                predict = pd.DataFrame(
                    data=predict[:, :len(endog_target_names)],
                    columns=endog_target_names)

            if self.pipeline_specification['model']['strategy'].startswith('SARIMAX'):
                all = self.problem_specification['targets'] + self.problem_specification['predictors']
                all = [item for item in all if item not in cross_section_names]
                exog_names = [i for i in self.problem_specification.get('exogenous', []) if i in all]
                endog = [i for i in all if i not in exog_names and i != time_name]

                if 'test' == forecast_mode:
                    predict = pd.DataFrame(
                        data=model.predict(len(model.model._index) - 1,
                                           len(model.model._index) + min(forecast_len, len(treatment.index)) - 1),
                        columns=endog
                    )
                elif 'train' == forecast_mode:
                    predict = pd.DataFrame(
                        data=model.predict(0, min(forecast_len, len(treatment.index)) - 1),
                        columns=endog
                    )
                else:
                    # All Split
                    predict = pd.DataFrame(
                        data=model.predict(0, len(treatment.index) - 1),
                        columns=endog
                    )

            if predict is not None:
                # Fix cross-section-name
                predict.reset_index(drop=True, inplace=True)
                predict[index_names] = index

                for i, section in enumerate(cross_section_names):
                    predict[section] = treatment_name[i]
                predictions.append(predict)

        if not predictions:
            return pd.DataFrame(data=[], columns=[index_names[0], *target_names])

        predictions = pd.concat(predictions)

        # remove interpolated entries in the time series that cannot be matched back to the input dataframe
        predictions.dropna(inplace=True)

        # cast d3mIndex back to int (it became float due to na values)
        predictions[index_names[0]] = predictions[index_names[0]].astype(int)

        # indices are duplicated after concat
        predictions.reset_index(drop=True, inplace=True)

        return predictions

    def refit(self, dataframe=None, data_specification=None):
        pass

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
                with warnings.catch_warnings():
                    joblib.dump(
                        self.preprocessors[treatment_name][preprocessor_name],
                        os.path.join(preprocess_treatment_dir, f'{preprocessor_name}.joblib'))

        model_dir = os.path.join(solution_dir, model_folder)
        os.makedirs(model_dir, exist_ok=True)
        for treatment_name in self.model:
            with warnings.catch_warnings():
                joblib.dump(
                    self.model[treatment_name],
                    os.path.join(solution_dir, model_folder, f'{hash(treatment_name)}.joblib'))

        with open(os.path.join(solution_dir, 'model_treatments.json'), 'w') as treatment_file:
            json.dump(
                [{'name': treatment_name, 'id': hash(treatment_name)} for treatment_name in self.model],
                treatment_file)
        # print('SAVED:', len(self.model))
        # print(solution_dir)

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

        # print('solution dir', solution_dir)

        if not metadata:
            metadata_path = os.path.join(solution_dir, 'solution.json')
            with open(metadata_path, 'r') as metadata_path:
                metadata = json.load(metadata_path)

        pipeline_specification = metadata['pipeline_specification']
        problem_specification = metadata['problem_specification']
        data_specification = metadata.get('data_specification')

        model_dir = os.path.join(solution_dir, metadata['model_folder'])
        preprocess_dir = os.path.join(solution_dir, metadata['preprocess_folder'])

        with open(os.path.join(solution_dir, 'model_treatments.json'), 'r') as treatment_file:
            treatments = json.load(treatment_file)

        preprocessors = {}
        models = {}
        for treatment in treatments:

            if type(treatment['name']) is list:
                treatment['name'] = tuple(treatment['name'])
            if type(treatment['name']) is not tuple:
                treatment['name'] = (treatment['name'],)

            model_path = os.path.join(model_dir, f'{treatment["id"]}.joblib')
            models[treatment['name']] = joblib.load(model_path)

            preprocess_path = os.path.join(preprocess_dir, str(treatment['id']))
            preprocessors[treatment['name']] = {}
            for preprocessor_name in os.listdir(preprocess_path):
                preprocessor_path = os.path.join(preprocess_path, preprocessor_name)
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


# class TorchModelsWrapper(BaseModelWrapper):
#
#     library = 'pytorch'
#
#     def describe(self):
#         if 1 == self.model.out_dim:
#             return {
#                 'model': 'AR_NN',
#                 'description': 'Auto-regressive model'
#             }
#
#         if 1 != self.model.out_dim:
#             return {
#                 'model': 'VAR_NN',
#                 'description': 'Vector Auto-regressive model'
#             }
#
#     def predict(self, dataframe):
#         cross_section_names = self.problem_specification.get('crossSection', [])
#         time_name = next(iter(self.problem_specification.get('time', [])), None)
#         index_names = self.problem_specification.get('indexes', ['d3mIndex'])
#         target_names = self.problem_specification['targets']
#
#         treatments_data = split_time_series(dataframe=dataframe, cross_section_names=cross_section_names)
#         predictions = []
#         predict = None
#
#         for treatment_name in treatments_data:
#             treatment = treatments_data[treatment_name]
#             if type(treatment_name) is not tuple:
#                 treatment_name = (treatment_name,)
#             if treatment_name not in self.model:
#                 print('unknown treatment:', treatment_name)
#                 continue
#             model = self.model[treatment_name]
#
#             treatment = format_dataframe_time_index(
#                 treatment,
#                 date=time_name,
#                 granularity_specification=self.problem_specification.get('timeGranularity'),
#                 freq=model.model._index.freq)
#             treatment.reset_index(inplace=True)
#
#             start = treatment[time_name].iloc[0]
#             end = treatment[time_name].iloc[-1]
#
#             index = treatment[index_names]
#             treatment.drop(index_names, inplace=True, axis=1)
#             # print('index', index)
#
#             if self.pipeline_specification['model']['strategy'] == 'AR':
#                 # model.model.endog = dataframe_history
#                 # standardize to dataframe
#                 predict = model.predict(
#                     # don't predict before autoregressive terms are populated
#                     start=max(start, model.model._index[model.model.k_ar]),
#                     # include the end date by offsetting end by one
#                     end=end)\
#                     .to_frame(name=self.problem_specification['targets'][0])
#
#                 if model.model._index[model.model.k_ar] > start:
#                     predict = pd.concat([
#                         pd.DataFrame(index=pd.date_range(
#                             start, model.model._index[model.model.k_ar - 1],
#                             freq=model.model._index.freq)),
#                         predict
#                     ])
#                 # index name is missing
#                 predict.index.name = time_name
#                 predict.reset_index()
#
#             if self.pipeline_specification['model']['strategy'] == 'VAR':
#                 endog_target_names = [i for i in self.problem_specification['targets'] if i != time_name]
#
#                 start_model = min(max(start, model.model._index[model.k_ar]), model.model._index[-1])
#                 end_model = max(start_model, end)
#
#                 new_time_index = pd.date_range(
#                     start=start_model,
#                     end=end_model,
#                     freq=model.model._index.freq)
#
#                 predict = model.model.predict(
#                     params=model.params,
#                     start=start_model, end=end_model)
#
#                 if len(predict) == 0:
#                     predict = np.empty(shape=(0, len(endog_target_names)))
#                 if len(predict) > len(new_time_index):
#                     predict = predict[:len(new_time_index)]
#
#                 # predictions don't provide dates; dates reconstructed based on freq
#                 predict = pd.DataFrame(
#                     data=predict[:, :len(endog_target_names)],
#                     columns=endog_target_names)
#
#                 # print('date range: ', start, end)
#                 # print('model range: ', start_model, end_model)
#                 predict[time_name] = new_time_index
#
#                 if start_model > start:
#                     # print('prepending data from before range')
#                     predict = pd.concat([
#                         pd.DataFrame(index=pd.date_range(
#                             start, model.model._index[model.k_ar - 1],
#                             freq=model.model._index.freq)),
#                         predict
#                     ])
#                 if start_model < start:
#                     # print(predict)
#                     # print('removing data from below the asked-for-interval, but was forecasted')
#                     predict = predict[predict[time_name] >= start]
#                     # print(predict)
#
#         if not predictions:
#             return pd.DataFrame(data=[], columns=[index_names[0], *target_names])
#
#         predictions = pd.concat(predictions)
#
#         # remove interpolated entries in the time series that cannot be matched back to the input dataframe
#         predictions.dropna(inplace=True)
#         # predictions = predictions[[*index_names, *target_names, time_name, *cross_section_names]]
#
#         # cast d3mIndex back to int (it became float due to na values)
#         predictions[index_names[0]] = predictions[index_names[0]].astype(int)
#
#         # indices are duplicated after concat
#         predictions.reset_index(drop=True, inplace=True)
#         return predictions
#
#     def refit(self, dataframe=None, data_specification=None):
#         raise NotImplementedError('Unsupported operation')
#
#     def save(self, solution_dir):
#         # self.model.remove_data()
#         os.makedirs(solution_dir, exist_ok=True)
#
#         model_folder = 'models'
#         preprocess_folder = 'preprocessors'
#
#         preprocess_dir = os.path.join(solution_dir, preprocess_folder)
#         for treatment_name in self.preprocessors:
#             for preprocessor_name in self.preprocessors[treatment_name]:
#                 preprocess_treatment_dir = os.path.join(preprocess_dir, str(hash(treatment_name)))
#                 os.makedirs(preprocess_treatment_dir, exist_ok=True)
#                 with warnings.catch_warnings():
#                     joblib.dump(
#                         self.preprocessors[treatment_name][preprocessor_name],
#                         os.path.join(preprocess_treatment_dir, f'{preprocessor_name}.joblib'))
#
#         model_dir = os.path.join(solution_dir, model_folder)
#         os.makedirs(model_dir, exist_ok=True)
#
#         for treatment_name in self.model:
#             with warnings.catch_warnings():
#                 torch.save(self.model[treatment_name],
#                            os.path.join(solution_dir, model_folder, f'{hash(treatment_name)}.pt'))
#
#         with open(os.path.join(solution_dir, 'model_treatments.json'), 'w') as treatment_file:
#             json.dump(
#                 [{'name': treatment_name, 'id': hash(treatment_name)} for treatment_name in self.model],
#                 treatment_file)
#
#         metadata_path = os.path.join(solution_dir, 'solution.json')
#         with open(metadata_path, 'w') as metadata_file:
#             json.dump({
#                 'library': self.library,
#                 'pipeline_specification': self.pipeline_specification,
#                 'problem_specification': self.problem_specification,
#                 'data_specification': self.data_specification,
#                 'model_folder': model_folder,
#                 'preprocess_folder': preprocess_folder
#             }, metadata_file)
#
#     @staticmethod
#     def load(solution_dir, metadata=None):
#
#         # print('solution dir', solution_dir)
#
#         if not metadata:
#             metadata_path = os.path.join(solution_dir, 'solution.json')
#             with open(metadata_path, 'r') as metadata_path:
#                 metadata = json.load(metadata_path)
#
#         pipeline_specification = metadata['pipeline_specification']
#         problem_specification = metadata['problem_specification']
#         data_specification = metadata.get('data_specification')
#
#         model_dir = os.path.join(solution_dir, metadata['model_folder'])
#         preprocess_dir = os.path.join(solution_dir, metadata['preprocess_folder'])
#
#         with open(os.path.join(solution_dir, 'model_treatments.json'), 'r') as treatment_file:
#             treatments = json.load(treatment_file)
#
#         preprocessors = {}
#         models = {}
#         for treatment in treatments:
#
#             if type(treatment['name']) is list:
#                 treatment['name'] = tuple(treatment['name'])
#             if type(treatment['name']) is not tuple:
#                 treatment['name'] = (treatment['name'],)
#
#             model_path = os.path.join(model_dir, f'{treatment["id"]}.pt')
#             models[treatment['name']] = torch.load(model_path)
#
#             preprocess_path = os.path.join(preprocess_dir, str(treatment['id']))
#             preprocessors[treatment['name']] = {}
#             for preprocessor_name in os.listdir(preprocess_path):
#                 preprocessor_path = os.path.join(preprocess_path, preprocessor_name)
#                 preprocessors[treatment['name']][preprocessor_name] = joblib.load(preprocessor_path)
#
#         return TorchModelsWrapper(
#             pipeline_specification=pipeline_specification,
#             problem_specification=problem_specification,
#             data_specification=data_specification,
#             model=models,
#             preprocessors=preprocessors
#         )
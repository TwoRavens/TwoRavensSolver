import inspect
import json
import os
import warnings

from tworaven_solver.model import BaseModelWrapper
from tworaven_solver.utilities import split_time_series, filter_args, DEFAULT_INDEX, format_dataframe_time_index, \
    resample_dataframe_time_index, set_column_transformer_inverse_transform

import joblib
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = 'raise'


class SciKitLearnWrapper(BaseModelWrapper):
    library = 'scikit-learn'

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
            'all_parameters': get_params(self.model)
        }

    def predict(self, dataframe):
        index = dataframe.index
        dataframe = dataframe[self.problem_specification['predictors']]
        # categorical variables should be converted to string
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
        ordering_column = self.problem_specification.get("forecastingHorizon", {}).get('column')
        index_names = self.problem_specification.get('indexes', [DEFAULT_INDEX])

        target_names = self.problem_specification['targets']

        treatments_data = split_time_series(dataframe=dataframe, cross_section_names=cross_section_names)
        predictions = []

        for treatment_name in treatments_data:
            treatment = treatments_data[treatment_name]
            # treatment = treatment.head(forecast_len)
            if type(treatment_name) is not tuple:
                pass
            if treatment_name not in self.model:
                print('unknown treatment:', treatment_name)
                continue
            model = self.model[treatment_name]

            treatment = format_dataframe_time_index(
                treatment,
                time_format=self.problem_specification.get('time_format', {}).get(ordering_column),
                order_column=ordering_column)

            if self.pipeline_specification.get('preprocess', {}).get('resample'):
                treatment = resample_dataframe_time_index(
                    dataframe=treatment,
                    freq=model._index.freq)

            treatment = treatment.head(forecast_len).copy()

            index, date_index = treatment[index_names], treatment.index

            # filter to handle the case where index plays the role of time
            treatment.drop([x for x in index_names if x in treatment.columns],
                           inplace=True, axis=1)

            # TODO: CHECK
            if ordering_column in treatment.columns:
                # Dummy dateTimeIndex is activated, the original order-column should be dropped.
                treatment.drop(ordering_column, inplace=True, axis=1)
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

            elif self.pipeline_specification['model']['strategy'].startswith('TRA_'):
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
            else:
                raise ValueError(f"Unrecognized model strategy: {self.pipeline_specification['model']['strategy']}")

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

        # cast index back to int (it became float due to na values)
        # TODO: allow non-integer indexes
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

        num_numerical_features = None
        if self.preprocessors:
            preprocess_dir = os.path.join(solution_dir, preprocess_folder)
            os.makedirs(preprocess_dir, exist_ok=True)
            for name in self.preprocessors:
                # hack to make ColumnTransformer invertible
                if hasattr(self.preprocessors[name], 'num_numerical_features'):
                    num_numerical_features = self.preprocessors[name].num_numerical_features
                joblib.dump(self.preprocessors[name], os.path.join(preprocess_dir, f'{name}.joblib'))

        metadata_path = os.path.join(solution_dir, 'solution.json')
        with open(metadata_path, 'w') as metadata_file:
            json.dump({
                'library': self.library,
                'pipeline_specification': self.pipeline_specification,
                'problem_specification': self.problem_specification,
                'data_specification': self.data_specification,
                'model_filename': model_filename,
                'preprocess_folder': preprocess_folder,
                'num_numerical_features': num_numerical_features
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load(model_path)

            preprocessors = {}
            if os.path.exists(preprocess_dir):
                for filename in os.listdir(preprocess_dir):
                    # hack to make ColumnTransformer invertible
                    preprocessor = joblib.load(os.path.join(preprocess_dir, filename))
                    if metadata.get('num_numerical_features'):
                        set_column_transformer_inverse_transform(preprocessor, metadata['num_numerical_features'])
                    preprocessors[filename.replace('.joblib', '')] = preprocessor

        return SciKitLearnWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            data_specification=data_specification,
            model=model,
            preprocessors=preprocessors
        )
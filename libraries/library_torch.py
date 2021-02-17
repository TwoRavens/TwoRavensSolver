import json
import os
import warnings

from tworaven_solver.model import BaseModelWrapper
from tworaven_solver.utilities import split_time_series, format_dataframe_time_index, DEFAULT_INDEX, \
    resample_dataframe_time_index
import pandas as pd
import numpy as np
import joblib

import torch


class TorchModelsWrapper(BaseModelWrapper):

    library = 'pytorch'

    def describe(self):
        if 1 == self.model.out_dim:
            return {
                'model': 'AR_NN',
                'description': 'Auto-regressive model'
            }

        if 1 != self.model.out_dim:
            return {
                'model': 'VAR_NN',
                'description': 'Vector Auto-regressive model'
            }

    def predict(self, dataframe):
        cross_section_names = self.problem_specification.get('crossSection', [])
        time_name = next(iter(self.problem_specification.get('time', [])), None)
        index_names = self.problem_specification.get('indexes', [DEFAULT_INDEX])
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

            treatment = format_dataframe_time_index(
                treatment,
                order_column=time_name,
                time_format=self.problem_specification.get('time_format', {}).get(time_name))

            if self.pipeline_specification.get('preprocess', {}).get('resample'):
                treatment = resample_dataframe_time_index(
                    dataframe=treatment,
                    freq=model.model._index.freq)
            treatment.reset_index(inplace=True)

            start = treatment[time_name].iloc[0]
            end = treatment[time_name].iloc[-1]

            index = treatment[index_names]
            treatment.drop(index_names, inplace=True, axis=1)
            # print('index', index)

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

        if not predictions:
            return pd.DataFrame(data=[], columns=[index_names[0], *target_names])

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
        raise NotImplementedError('Unsupported operation')

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
                torch.save(self.model[treatment_name],
                           os.path.join(solution_dir, model_folder, f'{hash(treatment_name)}.pt'))

        with open(os.path.join(solution_dir, 'model_treatments.json'), 'w') as treatment_file:
            json.dump(
                [{'name': treatment_name, 'id': hash(treatment_name)} for treatment_name in self.model],
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

            model_path = os.path.join(model_dir, f'{treatment["id"]}.pt')
            models[treatment['name']] = torch.load(model_path)

            preprocess_path = os.path.join(preprocess_dir, str(treatment['id']))
            preprocessors[treatment['name']] = {}
            for preprocessor_name in os.listdir(preprocess_path):
                preprocessor_path = os.path.join(preprocess_path, preprocessor_name)
                preprocessors[treatment['name']][preprocessor_name] = joblib.load(preprocessor_path)

        return TorchModelsWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            data_specification=data_specification,
            model=models,
            preprocessors=preprocessors
        )
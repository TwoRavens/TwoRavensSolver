import json
import os

import warnings

import joblib
import pandas as pd

from tworaven_solver.model import BaseModelWrapper
from tworaven_solver.utilities import split_time_series, DEFAULT_INDEX, \
    format_dataframe_time_index, resample_dataframe_time_index
# import prophet


class ProphetWrapper(BaseModelWrapper):
    library = 'prophet'

    def describe(self):
        # TODO: pull hyperparams out of model
        return {
            'model': f'Prophet()',
            'description': 'Prophet model'
        }

    def fitted(self):
        pass

    def predict(self, exog_future):
        cross_section_names = self.problem_specification.get('crossSection', [])
        ordering_column = self.problem_specification.get("forecastingHorizon", {}).get('column')
        index_names = self.problem_specification.get('indexes', [DEFAULT_INDEX])
        target_names = self.problem_specification['targets']
        treatments_data = split_time_series(dataframe=exog_future, cross_section_names=cross_section_names)
        predictions = []

        for treatment_name in treatments_data:
            treatment = treatments_data[treatment_name]
            if type(treatment_name) is not tuple:
                pass
            if treatment_name not in self.model:
                # print('Unknown treatment:', treatment_name)
                # print('Available treatments:', [name for name in self.model])
                continue
            model = self.model[treatment_name]

            treatment = format_dataframe_time_index(
                treatment,
                order_column=ordering_column,
                time_format=self.problem_specification.get('time_format', {}).get(ordering_column))

            if self.pipeline_specification.get('preprocess', {}).get('resample'):
                treatment = resample_dataframe_time_index(
                    dataframe=treatment,
                    freq=model.model._index.freq)

            # move order column out into dataframe
            treatment.reset_index(inplace=True)

            # save indexes for later use (must happen after reset_index, because ordering may be same as indexes)
            index = treatment[index_names]

            # drop indexes (but keep orderings)
            treatment.drop([x for x in index_names if x in treatment.columns],
                           inplace=True, axis=1)

            predict = model.predict(treatment)

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

        # cast index back to int (it became float due to na values)
        # TODO: allow non-integer indexes
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

        return ProphetWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            data_specification=data_specification,
            model=models,
            preprocessors=preprocessors)

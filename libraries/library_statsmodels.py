import json
import os

import warnings

import joblib
import pandas as pd
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tworaven_solver.model import BaseModelWrapper
from tworaven_solver.utilities import split_time_series, DEFAULT_INDEX, \
    resample_dataframe_time_index, format_dataframe_time_index, set_column_transformer_inverse_transform


class StatsModelsWrapper(BaseModelWrapper):
    library = 'statsmodels'

    def describe(self):
        if type(self.model) is AutoRegResultsWrapper:
            return {
                'model': f'AutoReg({self.model.k_ar})',
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
                'description': f'Seasonal autoregressive integrated moving average with exogenous regressors. '
                               f'The three values indicate the number of AR parameters, number of differences, and number of MA parameters.'
            }

    def fitted_values(self):
        cross_section_names = self.problem_specification.get('crossSection', [])
        ordering_column = self.problem_specification.get("forecastingHorizon", {}).get('column')
        index_names = self.problem_specification.get('indexes', [DEFAULT_INDEX])
        target_names = self.problem_specification['targets']

        # targets cannot be exogenous, subset exogenous labels to the predictor set
        exogenous_names = [i for i in self.problem_specification.get('exogenous', []) if
                           i in self.problem_specification['predictors'] and
                           i not in cross_section_names]

        # target variables are not transformed, all other variables are transformed
        endog_non_target_names = [i for i in self.problem_specification['predictors'] if
                                  i not in exogenous_names and
                                  i != ordering_column and
                                  i not in cross_section_names]
        endog_target_names = [i for i in target_names if i != ordering_column]

        predictions = []
        for treatment_name in self.model:
            predict = None

            if self.pipeline_specification['model']['strategy'] == 'AUTOREG':
                predict = self.model[treatment_name].fittedvalues
            if self.pipeline_specification['model']['strategy'] == 'VAR':
                predict = self.model[treatment_name].fittedvalues
            if self.pipeline_specification['model']['strategy'] == 'SARIMAX':
                predict = self.model[treatment_name].fittedvalues

            preprocess_endog = self.preprocessors[treatment_name].get('endogenous')
            if preprocess_endog and endog_non_target_names:
                print('BEFORE', predict)
                predict.iloc[:, len(endog_target_names):] = preprocess_endog.inverse_transform(predict.iloc[:, len(endog_target_names):])
                print('AFTER', predict)

            if predict is not None:
                # Fix cross-section-name
                predict.reset_index(drop=True, inplace=True)
                # predict[index_names] = index

                for i, section in enumerate(cross_section_names):
                    predict[section] = treatment_name[i]
                predictions.append(predict)

        if not predictions:
            return pd.DataFrame(data=[], columns=[index_names[0], *target_names])

        predictions = pd.concat(predictions)

        # remove interpolated entries in the time series that cannot be matched back to the input dataframe
        predictions.dropna(inplace=True)

        # cast index back to int (it became float due to na values)
        # TODO: allow non-integer indexes
        # predictions[index_names[0]] = predictions[index_names[0]].astype(int)

        # indices are duplicated after concat
        # predictions.reset_index(drop=True, inplace=True)

        return predictions

    def predict(self, exog_future):
        """out-of-sample"""
        cross_section_names = self.problem_specification.get('crossSection', [])
        ordering_column = self.problem_specification.get("forecastingHorizon", {}).get('column')
        index_names = self.problem_specification.get('indexes', [DEFAULT_INDEX])
        target_names = self.problem_specification['targets']

        # targets cannot be exogenous, subset exogenous labels to the predictor set
        exogenous_names = [i for i in self.problem_specification.get('exogenous', []) if
                           i in self.problem_specification['predictors'] and
                           i not in cross_section_names]

        # target variables are not transformed, all other variables are transformed
        endog_non_target_names = [i for i in self.problem_specification['predictors'] if
                                  i not in exogenous_names and
                                  i != ordering_column and
                                  i not in cross_section_names]
        endog_target_names = [i for i in target_names if i != ordering_column]

        treatments_data = split_time_series(dataframe=exog_future, cross_section_names=cross_section_names)
        predictions = []

        for treatment_name in treatments_data:
            predict = None
            treatment = treatments_data[treatment_name]

            if type(treatment_name) is not tuple:
                treatment_name = (treatment_name,)
            if treatment_name not in self.model:
                print('unknown treatment:', treatment_name)
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

            preprocess_exog = self.preprocessors[treatment_name]['exogenous']
            if preprocess_exog:
                treatment[exogenous_names] = preprocess_exog.transform(treatment[exogenous_names])
            preprocess_endog = self.preprocessors[treatment_name]['endogenous']
            if preprocess_endog:
                treatment[endog_non_target_names] = preprocess_endog.transform(treatment[endog_non_target_names])

            # move order column out into dataframe
            treatment.reset_index(inplace=True)

            # save indexes for later use (must happen after reset_index, because ordering may be same as indexes)
            index = treatment[index_names]

            # drop indexes (but keep orderings)
            treatment.drop([x for x in index_names if x in treatment.columns],
                           inplace=True, axis=1)

            if self.pipeline_specification['model']['strategy'] == 'AUTOREG':
                predict = model.forecast(
                    steps=len(treatment),
                    exog=treatment
                ).to_frame(name=self.problem_specification['targets'][0])

                predict.index.name = ordering_column
                # predict.reset_index()

            if self.pipeline_specification['model']['strategy'] == 'VAR':
                print(treatment)
                predict = model.forecast(
                    # TODO
                    y=None,
                    steps=len(treatment),
                    exog_future=treatment)

                # endog_target_names = [i for i in self.problem_specification['targets'] if i != ordering_column]
                # if len(predict) == 0:
                #     predict = np.empty(shape=(0, len(endog_target_names)))
                #
                # # predictions don't provide dates; dates reconstructed based on freq
                # predict = pd.DataFrame(
                #     data=predict[:, :len(endog_target_names)],
                #     columns=endog_target_names)

            if self.pipeline_specification['model']['strategy'].startswith('SARIMAX'):
                all = self.problem_specification['targets'] + self.problem_specification['predictors']
                all = [item for item in all if item not in cross_section_names]
                exog_names = [i for i in self.problem_specification.get('exogenous', []) if i in all]
                endog = [i for i in all if i not in exog_names and i != ordering_column]

                predict = pd.DataFrame(
                    data=model.forecast(len(treatment), exog=treatment),
                    columns=endog)

            # inverse-transform auxiliary variables
            preprocess_endog = self.preprocessors[treatment_name]['endogenous']
            if preprocess_endog:
                treatment[endog_non_target_names] = preprocess_endog.inverse_transform(treatment[endog_non_target_names])

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
                preprocessor = joblib.load(preprocessor_path)
                if metadata.get('num_numerical_features'):
                    set_column_transformer_inverse_transform(preprocessor, metadata['num_numerical_features'])
                preprocessors[treatment['name']][preprocessor_name] = preprocessor

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
            preprocessors=preprocessors)
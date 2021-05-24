import copy

from tworaven_solver.solution import Solution, Estimator
import os
import json
import pandas as pd

from tworaven_solver.utilities import TrainSpecification


class CrossSectionalEstimator(Estimator):
    library = 'cross-sectional'

    def fitted_values(self):
        all_fitted_values = []
        for treatment_name in self.estimator:
            fitted_values = self.estimator[treatment_name].fitted_values()

            if fitted_values is not None:
                # Fix cross-section-name
                fitted_values.reset_index(drop=True, inplace=True)

                for i, section in enumerate(self.problem.cross_sections):
                    fitted_values[section] = treatment_name[i]
                all_fitted_values.append(fitted_values)

        if not all_fitted_values:
            return pd.DataFrame(data=[], columns=[self.problem.indexes[0], *self.problem.cross_sections, *self.problem.targets])

        return pd.concat(all_fitted_values)

    def fit(self, dataframe):
        dataframe_split = split_cross_sections(
            dataframe=dataframe,
            cross_section_names=self.train_specification.problem.cross_sections)

        inner_train_spec: TrainSpecification = copy.deepcopy(self.train_specification)
        del inner_train_spec.problem.spec['cross_sections']

        self.estimator = {}
        for treatment_name in dataframe_split:
            treatment_solution = Solution(
                pipeline_specification=self.model_specification['strategy'],
                train_specification=inner_train_spec)
            treatment_solution.fit(dataframe_split[treatment_name])
            self.estimator[treatment_name] = treatment_solution

        return self

    def predict(self, exog_future):
        treatments_data = split_cross_sections(dataframe=exog_future, cross_section_names=self.problem.cross_sections)
        predictions = []

        for treatment_name in treatments_data:
            if treatment_name not in self.estimator:
                print('unknown treatment:', treatment_name)
                continue

            predict = self.estimator[treatment_name].predict(treatments_data[treatment_name])

            if predict is not None:
                # Fix cross-section-name
                predict.reset_index(drop=True, inplace=True)

                for i, section in enumerate(self.problem.cross_sections):
                    predict[section] = treatment_name[i]
                predictions.append(predict)

        if not predictions:
            return pd.DataFrame(data=[], columns=[self.problem.indexes[0], *self.problem.cross_sections, *self.problem.targets])

        return pd.concat(predictions)

    def refit(self, dataframe=None, data_specification=None):
        pass

    def save(self, solution_dir):
        os.makedirs(solution_dir, exist_ok=True)

        models_dir = os.path.join(solution_dir, 'estimators')
        os.makedirs(models_dir, exist_ok=True)
        for treatment_name in self.estimator:
            treatment_dir = os.path.join(models_dir, str(hash(treatment_name)))
            self.estimator[treatment_name].save(treatment_dir)

        with open(os.path.join(solution_dir, 'model_treatments.json'), 'w') as treatment_file:
            json.dump(
                [{'name': treatment_name, 'id': hash(treatment_name)} for treatment_name in self.estimator],
                treatment_file)

    @staticmethod
    def _load_estimator(solution_dir):
        models_dir = os.path.join(solution_dir, 'estimators')

        with open(os.path.join(solution_dir, 'model_treatments.json'), 'r') as treatment_file:
            treatments = json.load(treatment_file)

        estimators = {}
        for treatment in treatments:
            if type(treatment['name']) is list:
                treatment['name'] = tuple(treatment['name'])
            if type(treatment['name']) is not tuple:
                treatment['name'] = (treatment['name'],)

            estimators[treatment['name']] = Solution.load(os.path.join(models_dir, str(treatment['id'])))

        return estimators


def split_cross_sections(dataframe, cross_section_names=None):
    """
    break a dataframe with cross sectional indicators into a dict of dataframes containing each treatment
    :param dataframe:
    :param cross_section_names: column names of cross sectional variables
    :return dict of {(cs1, cs2, ...): treatment_data, ...}
    """

    # avoid unecessary data re-allocation
    if not cross_section_names:
        return {(): dataframe}

    def to_tuple(v):
        if isinstance(v, tuple):
            return v
        return (v,)

    other_columns = [i for i in dataframe.columns.values if i not in cross_section_names]
    treatments = {to_tuple(label): data[other_columns] for label, data in dataframe.groupby(cross_section_names)}
    for treatment in treatments:
        treatments[treatment].reset_index(drop=True, inplace=True)
    return treatments

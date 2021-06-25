import os

import pandas as pd

import tworaven_solver
from tworaven_solver.transformers.time_series import standardize_date_offset


def filter_args(arguments, intersect):
    return {k: arguments[k] for k in intersect if k in arguments}


class Dataset(object):
    def __init__(self, input):
        if not input:
            raise ValueError('No input provided.')

        if 'resource_uri' not in input:
            raise ValueError('Invalid input: no resource_uri provided.')

        self.input = input

    def get_dataframe(self, **options):
        if 'delimiter' in self.input:
            options.setdefault('delimiter', self.input['delimiter'])

        return pd.read_csv(self.get_resource_uri(), **options)

    def get_resource_uri(self):
        return self.input['resource_uri']

    def get_file_path(self):
        return os.path.join(*self.get_resource_uri().replace('file://', '').split('/'))

    def get_name(self):
        return self.input.get('name', self.input['resource_uri'])


# I'm not subclassing dict because I want ProblemSpecification to be an adapter on the original dict
# This should handle all logic around strange values in json- duplicates, missing keys, overlapping disjoint sets, etc.
class ProblemSpecification(object):
    """Helper class for interpreting problem-specification jsons"""
    DEFAULT_INDEX = 'd3mIndex'
    SUPPORTED_TASK_TYPES = ['REGRESSION', 'CLASSIFICATION']

    def __init__(self, specification):
        if isinstance(specification, tworaven_solver.utilities.ProblemSpecification):
            specification = specification.spec
        self.spec: dict = specification

    def __getitem__(self, item):
        return self.spec[item]

    def __setitem__(self, key, item):
        self.spec[key] = item

    def get(self, key, default=None):
        return self.spec.get(key, default)

    @property
    def task(self):
        if all(i in self.categoricals for i in self.targets):
            return 'CLASSIFICATION'
        if not any(i in self.categoricals for i in self.targets):
            return 'REGRESSION'

    @property
    def is_forecasting(self):
        return self.spec.get('forecasting', False)

    @property
    def keywords(self):
        return [
            self.task,
            'MULTIVARIATE' if len(self.targets) > 1 else 'UNIVARIATE',
            'FORECASTING' if self.is_forecasting else 'PREDICTION',
            *self.get('keywords', [])
        ]

    def date_format(self, name):
        return self.spec.get('date_format', {}).get(name)

    def set_date_format(self, name, date_format):
        self.spec.setdefault('date_format', {})[name] = date_format

    def date_offset_start(self, name):
        return self.spec.get('date_offset_start', {}).get(name)

    def date_offset_unit(self, name):
        offset_unit = self.spec.get('date_offset_unit', {}).get(name)
        if offset_unit is not None:
            return standardize_date_offset(offset_unit)

    @property
    def resample_date_offset_unit(self):
        """time offset of resampled data"""
        granularity = self.spec.get('timeGranularity', {})
        units = granularity.get('units')
        if units:
            return granularity.get('value', 1) * standardize_date_offset(units)

    @property
    def ordering(self):
        if not self.is_forecasting:
            return

        order_name = self.spec.get('forecastingHorizon', {}).get('column')
        if order_name is not None:
            return order_name

        # fall back to indexes
        if len(self.indexes) > 1:
            raise ValueError("Ordering is indeterminate. No forecasting column given, and multiple indexes given")
        order_name = self.indexes[0]
        if order_name in self.categoricals:
            raise ValueError("Ordering is indeterminate. No forecasting column given, and index is categorical")
        return order_name

    def set_ordering(self, name):
        self.spec.setdefault('forecastingHorizon', {})
        self.spec['forecastingHorizon']['column'] = name

    @property
    def weighting(self):
        if self.spec.get('weights'):
            return self.spec['weights'][0]

    @property
    def indexes(self):
        # if data has null or empty indexes, let DEFAULT_INDEX be the only index
        self.spec.setdefault('indexes', [self.DEFAULT_INDEX])
        if len(self.spec['indexes']) == 0:
            self.spec['indexes'] = [self.DEFAULT_INDEX]

        return self.spec['indexes']

    @property
    def cross_sections(self):
        return self.spec.get('crossSection', [])

    @property
    def categoricals(self):
        categorical = self.spec.get('categorical', [])
        if not self.is_forecasting:
            categorical += self.cross_sections
        return list(set(categorical))

    @property
    def predictors(self):
        """get deduped, non-null, non-tagged variables from the predictors list"""
        predictors = [i for i in set(self.spec.get('predictors', [])) if i is not None]
        if self.weighting in predictors:
            predictors.remove(self.weighting)
        if self.ordering in predictors:
            predictors.remove(self.ordering)
        # if forecasting, then cross sectional variables are separate
        # otherwise, cross sectional variables are categorical predictors
        if self.is_forecasting:
            for cross_sectional in self.cross_sections:
                if cross_sectional in predictors:
                    predictors.remove(cross_sectional)
        else:
            predictors.extend(self.cross_sections)
        return list(set(predictors))

    @property
    def exogenous(self):
        return [
            i for i in self.spec.get('exogenous', []) if
            i not in self.cross_sections
        ]

    @property
    def endogenous(self):
        exogenous = self.exogenous
        return [
            i for i in self.predictors if
            i not in self.cross_sections and
            i not in exogenous
        ]

    @property
    def targets(self):
        return self.spec.get('targets', [])

    @property
    def y(self):
        # model outputs
        names = [*self.targets]
        if self.ordering:
            names.append(self.ordering)

        return list(set(names))

    @property
    def X(self):
        # model inputs use the same columns as the test data
        return self.train

    @property
    def test(self):
        names = [*self.indexes, *self.predictors]
        if self.ordering:
            names.append(self.ordering)
        if self.cross_sections:
            names.extend(self.cross_sections)

        return list(set(names))

    @property
    def train(self):
        names = [*self.indexes, *self.predictors, *self.targets]
        if self.ordering:
            names.append(self.ordering)
        if self.weighting:
            names.append(self.weighting)
        if self.cross_sections:
            names.extend(self.cross_sections)

        return list(set(names))


class TrainSpecification(object):
    """setters and getters for attributes in train specifications"""

    def __init__(self, specification):
        if isinstance(specification, TrainSpecification):
            specification = specification.spec
        self.spec = specification

    def __getitem__(self, item):
        return self.spec[item]

    @property
    def problem(self):
        return ProblemSpecification(self.spec['problem'])

    @property
    def performance_metric(self):
        return self.spec.get("performanceMetric", {'metric': "MEAN_SQUARED_ERROR"})

    def dataset(self, name):
        if name not in self.spec:
            raise ValueError(f'unrecognized dataset name: {name}')
        return Dataset(self.spec[name])


from typing import List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tworaven_solver.transformers.time_series import TemporalPreprocessor, format_dataframe_date_index, TemporalImputer
from tworaven_solver.utilities import ProblemSpecification

pd.options.mode.chained_assignment = 'raise'


class InvertibleSimpleImputer(SimpleImputer):
    """why bother trying to stick NaN's back in?"""
    def inverse_transform(self, X):
        return X


class ColumnSelector(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        if y is not None:
            raise ValueError('y must be None')
        return self

    def transform(self, X, y=None):
        if y is not None:
            raise ValueError('y must be None')
        # copy to ensure that this transform is not returning a view
        return X[[col for col in self.columns if col in X.columns]].copy()

    def inverse_transform(self, X, y=None):
        return X


class PandasStandardScaler(TransformerMixin):
    def __init__(self, indexes=None):
        self.indexes = indexes or []
        self.mean = None
        self.variance = None

    def fit(self, X, y=None):
        if y is not None:
            raise ValueError('y must be None')

        self.mean = X.mean(axis=0)
        self.variance = X.var(axis=0)
        return self

    def transform(self, X, y=None):
        if y is not None:
            raise ValueError('y must be None')
        return (X - self.mean) / self.variance

    def inverse_transform(self, X, y=None):
        return X * self.variance + self.mean


class DropConstant(TransformerMixin):
    def __init__(self):
        super().__init__()
        self.mask = None
        self.columns = None
        self.values = None

    def fit(self, X, y=None):
        if y is not None:
            raise ValueError('y must be None')
        if isinstance(X, pd.DataFrame):
            self.mask = (X != X.iloc[0]).any()
            self.columns = [col for col, mask in zip(X.columns, self.mask) if not mask]
            self.values = X.loc[:, self.mask].iloc[0]
        elif isinstance(X, np.ndarray):
            self.mask = np.any(X != X[0, :], axis=0)
            self.values = X[0, self.mask]
        else:
            raise ValueError("unknown type", type(X))
        return self

    def transform(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.mask]
        if isinstance(X, np.ndarray):
            return X[:, self.mask]
        else:
            raise ValueError("unknown type", type(X))

    def inverse_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for (i, mask), colname, val in zip(enumerate(self.mask), self.columns, self.values):
                if mask:
                    X.insert(i, column=colname, value=val)
            return X
        elif isinstance(X, np.array):
            raise NotImplementedError
        else:
            raise ValueError("unknown type", type(X))


class NamedPipeline(Pipeline):
    def get_feature_names(self, names):
        for step in self.steps:
            names = step.get_feature_names(names)
        return names


class InvertibleColumnTransformer(ColumnTransformer):
    """Extension of ColumnTransformer to add inverse_transform"""

    def _get_transformer_column_counts(self) -> List[int]:
        """
        Get number of columns emitted by each transformer.
        :returns col_counts: Number of the features produced by each transform.
        """
        col_counts = []
        for name, trans, column, _ in self._iter(fitted=True):
            if trans == 'drop' or len(column) == 0:
                col_counts.append(0)
            elif trans == 'passthrough':
                col_counts.append(len(column))
            elif isinstance(trans, StandardScaler):
                col_counts.append(len(trans.mean_))
            elif not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not "
                                     "provide get_feature_names."
                                     % (str(name), type(trans).__name__))
            else:
                # TODO: fix this! yikes
                col_counts.append(len(column))
                # col_counts.append(trans.get_feature_names())
        return col_counts

    def inverse_transform(self, X):
        """
        1. zip the transformers with the subset of columns that transformer was applied on
        2. invert the columns
        3. column stack the inverted data
        :param X: dataframe from output space of Self
        :return: corresponding dataframe from input space of Self
        """
        inverses = []
        col_offset = 0
        for (_name, transformer, colnames), num_columns in zip(self.transformers_,
                                                               self._get_transformer_column_counts()):
            if num_columns == 0:
                continue
            inverse = transformer.inverse_transform(X.iloc[:, col_offset:col_offset + num_columns])
            inverses.append(pd.DataFrame(inverse, columns=colnames))
            col_offset += num_columns
        return pd.concat(inverses, axis=1) if inverses else X


class ProblemPreprocessor(TransformerMixin):
    """general transformer that encapsulates all preprocessing transformations for problems"""
    def __init__(self, problem: ProblemSpecification, preprocess_specification: dict = None):
        self.problem = problem
        self.spec = preprocess_specification or {}
        self.transformer_X = None
        self.transformer_y = None

    def fit(self, X, y=None):
        """
        :param X: the entire dataframe-- all data
        :param y: must be empty
        :return: a fitted ProblemPreprocessor
        """
        if y is not None:
            raise ValueError("y must be None")

        # drop rows with nulls in the target columns
        existent_targets = [tar for tar in self.problem.targets if tar in X.columns]
        if existent_targets:
            X = X[X[existent_targets].notnull().all(1)]

        # alternative pipeline for time series problems
        if self.problem.is_forecasting:
            self.transformer_y = NamedPipeline(steps=[
                ('selector', ColumnSelector(columns=self.problem.y)),
                ('temporal', TemporalPreprocessor(
                    order_column=self.problem.ordering,
                    indexes=self.problem.indexes,
                    date_format=self.problem.date_format(self.problem.ordering),
                    date_offset_start=self.problem.date_offset_start(self.problem.ordering),
                    input_date_offset_unit=self.problem.date_offset_unit(self.problem.ordering),
                    resample=self.spec.get('resample'),
                    resample_date_offset_unit=self.problem.resample_date_offset_unit)),
                ('constant', DropConstant()),
                ('imputer', TemporalImputer()),
                ('scaler', PandasStandardScaler())
            ])
            self.transformer_y.fit(X)

            if self.problem.exogenous:
                # re-use fitted params from transformer_X to ensure that params are consistent
                self.transformer_X = NamedPipeline(steps=[
                    ('selector', ColumnSelector(columns=self.problem.X)),
                    ('temporal', TemporalPreprocessor(**self.transformer_y['temporal'].get_params())),
                    ('constant', DropConstant()),
                    ('imputer', TemporalImputer()),
                    ('scaler', PandasStandardScaler())
                ])
                self.transformer_X.fit(X)

        # standard pipeline for ml problems
        else:
            self.transformer_X = NamedPipeline(steps=[
                ('selector', ColumnSelector(columns=self.problem.X)),
                ('preprocess', self._make_predict_transform(X[self.problem.X]))
            ])
            self.transformer_X.fit(X)
            self.transformer_y = NamedPipeline(steps=[
                ('selector', ColumnSelector(columns=self.problem.targets)),
                ('preprocess', self._make_predict_transform(X[self.problem.targets]))
            ])
            self.transformer_y.fit(X)

        return self

    def _make_predict_transform(self, dataframe: pd.DataFrame):
        # list columns that must be categorical, according to the datatype
        categorical_dtype_features = list(
            dataframe.select_dtypes(exclude=[np.number, "bool_", "object_"]).columns.values)
        # union categorical dtype features with features the user has explicitly labeled as categorical
        categorical_features = [i for i in set(self.problem.categoricals + categorical_dtype_features) if
                                i in dataframe.columns]

        # keep up to the 20 most frequent levels
        category_limit = self.spec.get('category_limit', 20)
        categories = [dataframe[col].value_counts()[:category_limit].index.tolist() for col in categorical_features]

        numerical_features = [i for i in dataframe.columns.values if i not in categorical_features]
        numerical_transformer = NamedPipeline(steps=[
            ('imputer', InvertibleSimpleImputer(strategy='median')),
            ('scaler', PandasStandardScaler())
        ])

        categorical_transformer = NamedPipeline(steps=[
            ('imputer', InvertibleSimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False))
        ])

        # preprocess every column in the dataframe in an invertible transformation
        return InvertibleColumnTransformer(transformers=[
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)
        ])

    def transform(self, X, y=None):
        if y is not None:
            raise ValueError("y must be None")
        if X is None:
            raise ValueError("X must not be None")

        # drop rows with nulls in the target columns
        existent_targets = [tar for tar in self.problem.targets if tar in X.columns]
        if existent_targets:
            X = X[X[existent_targets].notnull().all(1)]

        output = {}
        if self.transformer_X:
            X_out = self.transformer_X.transform(X)
            # TODO: X_out is NOT a dataframe
            if len(X_out) == len(X):
                output['indexes'] = X[self.problem.indexes].copy()
            # if self.problem.weighting in X_out:
            #     output['weighting'] = X_out.pop(self.problem.weighting)
            output['X'] = X_out
        if self.transformer_y:
            y_out = self.transformer_y.transform(X)
            if len(y_out) == len(X):
                output['indexes'] = X[self.problem.indexes].copy()
            # if set(self.problem.indexes).issubset(set(y_out.columns.values)):
            #     output['indexes'] = y_out[self.problem.indexes]
            #     y_out.drop(columns=self.problem.indexes, inplace=True)
            output['y'] = y_out

        return output

    def inverse_transform(self, X=None, y=None):
        # standardize input to a dict of keys
        data = X
        if data is None:
            data = {}
        if y is not None:
            data['y'] = y

        # inverse the transform
        X_in = data.get('X')
        if X_in is not None and self.transformer_X:
            if 'indexes' in data:
                data['X'][self.problem.indexes] = data['indexes']
            if 'weighting' in data:
                data['X'][self.problem.weighting] = data['weighting']
            X_in = self.transformer_X.inverse_transform(data['X'])

        y_in = data.get('y')
        if y_in is not None and self.transformer_y:
            if 'indexes' in data:
                data['y'][self.problem.indexes] = data['indexes']
            y_in = self.transformer_y.inverse_transform(data['y'])

        # join the dict back together
        if X_in is not None and y_in is not None:
            data = pd.concat([X_in, y_in[y_in.columns.difference(X_in.columns)]], axis='columns')
        elif X_in is not None:
            data = X_in
        elif y_in is not None:
            data = y_in
        return data

    def format_dataframe_date_index(self, data: pd.DataFrame):
        """
        helper function to only format the date index, using the fitted preprocessor's params
        :param data: dataframe to be formatted in-place
        :return: Optional[pd.DateOffset] observation frequency
        """
        if not self.problem.is_forecasting:
            raise NotImplementedError("format_dataframe_date_index is only applicable for time series problems")

        trans: TemporalPreprocessor = self.transformer_y['temporal']
        return format_dataframe_date_index(
            data, trans.order_column,
            indexes=trans.indexes,
            date_format=trans.date_format,
            date_offset_unit=trans.input_date_offset_unit,
            date_offset_start=trans.date_offset_start)


# TODO:
# 2. decompose time series
# 3. preserve indexes
# 4. test degenerate cases- zero columns, zero rows
# 5. time series data should always be float
# 6. handle categoricals in time series

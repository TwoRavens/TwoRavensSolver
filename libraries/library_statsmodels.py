import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tworaven_solver.solution import Estimator
from tworaven_solver.utilities import filter_args


class StatsModelsEstimator(Estimator):
    library = 'statsmodels'

    def describe(self):
        if isinstance(self.estimator, AutoRegResultsWrapper):
            return {
                'model': f'AutoReg({self.estimator.k_ar})',
                'description': 'Autoregressive model'
            }

        if isinstance(self.estimator, VARResultsWrapper):
            return {
                'model': f'VAR({self.estimator.k_ar})',
                'description': 'Vector Autoregressive model'
            }

        if isinstance(self.estimator, SARIMAXResultsWrapper):
            return {
                'model': f'SARIMAX({self.estimator.model.k_ar}, {self.estimator.model.k_diff}, {self.estimator.model.k_ma})',
                'description': f'Seasonal autoregressive integrated moving average with exogenous regressors. '
                               f'The three values indicate the number of AR parameters, number of differences, and number of MA parameters.'
            }

    def fitted_values(self):
        return self.estimator.fittedvalues.to_frame(name=self.problem.targets[0])

    def fit(self, data):
        self.estimator = {
            'AR': fit_model_ar,
            'VAR': fit_model_var,
            'SARIMAX': fit_model_sarimax
        }[self.model_specification['strategy']](data, self.model_specification)
        return self

    def predict(self, data):
        steps = len(data['y'])
        exog_future = data.get('X')

        strategy = self.model_specification['strategy']
        if strategy == 'AR':
            # forecast doesn't have exog_oos, but predict does. forecast is just a limited wrapper for predict:
            # https://github.com/statsmodels/statsmodels/blob/3d3207122797e959a5f1dea4563b68b46bfa49ff/statsmodels/tsa/ar_model.py#L1333-L1337
            start = self.estimator.model.data.orig_endog.shape[0]
            predict = self.estimator.predict(
                start=start, end=start + steps - 1, dynamic=False,
                exog_oos=exog_future if self.problem.exogenous else None
            ).to_frame(name=self.problem.targets[0])

        elif strategy == 'VAR':
            predict = self.estimator.forecast(
                # TODO
                y=None,
                steps=steps,
                exog_future=exog_future)

            # endog_target_names = [i for i in self.problem_specification['targets'] if i != ordering_column]
            # if len(predict) == 0:
            #     predict = np.empty(shape=(0, len(endog_target_names)))
            #
            # # predictions don't provide dates; dates reconstructed based on freq
            # predict = pd.DataFrame(
            #     data=predict[:, :len(endog_target_names)],
            #     columns=endog_target_names)

        elif strategy == 'SARIMAX':
            predict = self.estimator.forecast(steps=steps, exog=exog_future)\
                .to_frame(name=self.problem.targets[0])

        else:
            raise ValueError(f'unrecognized strategy: {strategy}')

        # re-add the index
        if exog_future and 'indexes' in data:
            predict[self.problem.indexes] = exog_future['indexes']

        return predict

    def refit(self, dataframe=None, data_specification=None):
        pass


def fit_model_ar(data, model_specification):
    """
    Return a fitted autoregression model

    :param data:
    :param model_specification: {'lags': int, ...}
    """

    from statsmodels.tsa.ar_model import AutoReg
    return AutoReg(
        endog=data['y'],
        exog=data.get('X'),
        **filter_args(model_specification, ['lags', 'trend', 'seasonal', 'hold_back', 'period'])
    ).fit()


def fit_model_var(data, model_specification):
    """
    Return a fitted vector autoregression model

    :param data:
    :param model_specification: {'lags': int, ...}
    """

    from statsmodels.tsa.vector_ar.var_model import VAR

    model = VAR(
        endog=data['y'],
        exog=data.get('X'),
        freq=data['y'].index.freq)
    # VAR cannot be trained with start_params, while AR can
    return model.fit(**filter_args(model_specification, ['maxlags', 'ic', 'trend']))


def fit_model_sarimax(data, model_specification):
    """
    Return a fitted autoregression model

    :param data:
    :param model_specification:
    """

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    exog = data.get('X')
    if exog is not None:
        exog = exog.astype(float)

    model = SARIMAX(
        endog=data['y'].astype(float),
        exog=exog,
        # freq=freq,
        **filter_args(model_specification, [
            "order", "seasonal_order", "trend", "measurement_error",
            "time_varying_regression", "mle_regression", "simple_differencing",
            "enforce_stationarity", "enforce_invertibility", "hamilton_representation",
            "concentrate_scale", "trend_offset", "use_exact_diffuse"]))

    return model.fit(**filter_args(model_specification, [
        "start_params", "transformed", "includes_fixed", "cov_type", "cov_kwds",
        "method", "maxiter", "full_output", "disp", "callback", "return_params",
        "optim_score", "optim_complex_step", "optim_hessian", "flags", "low_memory"]))

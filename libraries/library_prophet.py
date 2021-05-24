import pandas as pd

from tworaven_solver.solution import Estimator
from prophet import Prophet

from utilities import filter_args


class ProphetEstimator(Estimator):
    library = 'prophet'

    def describe(self):
        # TODO: pull hyperparams out of model
        return {
            'model': f'Prophet()',
            'description': 'Prophet model'
        }

    def fit(self, data):

        from prophet import Prophet

        self.estimator = Prophet(**filter_args(self.model_specification, [
            'growth', 'changepoints', 'n_changepoints', 'changepoint_range',
            'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
            'holidays', 'seasonality_mode', 'seasonality_prior_scale',
            'holidays_prior_scale', 'changepoint_prior_scale',
            'mcmc_samples', 'interval_width', 'uncertainty_samples']))

        self.estimator.fit(self._format_input(data))
        return self

    def fitted_values(self):
        return self._format_output(self.estimator.predict())

    def predict(self, exog_future):
        future = self.estimator.make_future_dataframe(
            periods=len(exog_future),
            freq=pd.to_datetime(exog_future).index.freq)
        return self._format_output(self.estimator.predict(future))

    @staticmethod
    def _format_input(data):
        assert len(data['y'].columns) == 1, 'Prophet only supports univariate time series'
        data['y'].columns = ['y']
        data['y'].index.name = 'ds'
        data['y'].reset_index()
        return data['y']

    def _format_output(self, dataframe):
        return pd.DataFrame(dataframe[['y']], columns=self.problem.targets[0], index=dataframe['ds'])

    def refit(self, dataframe=None, data_specification=None):
        self.estimator = Prophet().fit(dataframe, init=stan_init(self.estimator))
        return self


# https://facebook.github.io/prophet/docs/additional_topics.html#updating-fitted-models
def stan_init(m):
    """Retrieve parameters from a trained model.

    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.

    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

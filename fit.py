import inspect

import pandas as pd
from .utilities import TrainSpecification, ProblemSpecification, filter_args

pd.options.mode.chained_assignment = 'raise'


_LOSS_FUNCTIONS = {
    'MEAN_SQUARED_ERROR': 'squared_loss',
    'R_SQUARED': 'rooted_squared_loss',
    'MEAN_ABSOLUTE_ERROR': 'mean_absolute_loss',
}


def fit_model(dataframe, model_specification, problem_specification, start_params=None):
    if model_specification['strategy'] == 'AR':
        model_specification = {
            'start_params': start_params,
            **model_specification
        }
        
    return {
        'VAR_NN': fit_model_var_ann,
        'PROPHET': fit_model_prophet,
    }[model_specification['strategy']](dataframe, model_specification, problem_specification)



def fit_model_var_ann(dataframes, model_specification, problem: ProblemSpecification):
    """
    Return a fitted autoregression neural network model
    :param dataframes: ordered by time index
    :param model_specification: {'lags': int, ...}
    :param problem:
    """
    # 'Y' variable is in the first column, AR only requires 'Y' value
    back_steps = model_specification.get('back_steps', 1)  # At least 1 time step is required
    loss_func = problem.get('performanceMetric').get('metric')
    loss_func = 'MEAN_SQUARED_ERROR' if (not loss_func or loss_func not in _LOSS_FUNCTIONS) else loss_func

    # Only considering endogenous features now
    if dataframes.get('X'):
        print('Exogenous features will not be considered now.')

    endog_mask = dataframes['y'].var(axis=0) > 0
    tgt_y = dataframes['y'][endog_mask.index[endog_mask]].astype(float)
    tgt_x = dataframes['X']

    # Build training matrix, can be moved to a new auxiliary function
    for step in range(1, back_steps + 1):
        tgt_y = tgt_y.drop(tgt_y.index[0])
        tmp_x = tgt_x.shift(step)
        tmp_x.columns = ['{}_minus_{}'.format(col, step) for col in tmp_x.columns]
        tgt_x = pd.concat((tgt_x, tmp_x), axis=1)

    train_x, train_y = tgt_x.dropna(), tgt_y.dropna()

    from .nn_models.NlayerMLP import ModMLPForecaster

    # Training model for current df
    model = ModMLPForecaster(loss=_LOSS_FUNCTIONS[loss_func])
    model.fit(train_x, train_y)

    # history points should be stored for future inference
    model.set_history(tgt_y.tail(back_steps), dataframes['time'])

    return model

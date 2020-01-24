from .utilities import Dataset, preprocess

from .utilities import (
    filter_args,
    get_freq,
    format_dataframe_time_index
)
import pandas as pd


# given a pipeline json and data, return a solution
def fit_pipeline(pipeline_specification, train_specification):
    # 1. load data
    dataframe = Dataset(train_specification['input']).get_dataframe()
    problem_specification = train_specification['problem']

    if problem_specification['taskType'] == 'FORECASTING':
        time = next(iter(train_specification['problem'].get('time', [])), None)
        dataframe = format_dataframe_time_index(dataframe, date=time)

    weights = problem_specification.get('weights')
    if weights and weights[0] in problem_specification['predictors']:
        problem_specification['predictors'].remove(weights[0])

    times = problem_specification.get('time')
    if times and times[0] in problem_specification['predictors']:
        problem_specification['predictors'].remove(times[0])

    stimulus, preprocessor = fit_preprocess(
        dataframe[problem_specification['predictors']],
        pipeline_specification['preprocess'],
        train_specification)

    dataframes = {
        'targets': dataframe[problem_specification['targets']] if problem_specification['targets'] else None,
        'predictors': pd.DataFrame(data=stimulus, index=dataframe.index) if problem_specification['predictors'] else None,
        'weight': dataframe[weights[0]] if weights else None
    }

    # 3. modeling
    model_specification = pipeline_specification['model']
    model = fit_model(dataframes, model_specification, problem_specification)

    # 4. wrap and save
    from .model import StatsModelsWrapper, SciKitLearnWrapper
    if model_specification['strategy'] in ['AR', 'SARIMAX', 'VAR']:
        return StatsModelsWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            model=model,
            preprocess=preprocessor)

    if model_specification['strategy'] in [
        'ORDINARY_LEAST_SQUARES', 'LOGISTIC_REGRESSION', 'RANDOM_FOREST', 'SUPPORT_VECTOR_CLASSIFIER'
    ]:
        return SciKitLearnWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            model=model,
            preprocess=preprocessor)


def fit_preprocess(dataframe, preprocess_specification, train_specification):
    # TODO: more varied preprocessing based on preprocess specification
    return preprocess(dataframe, train_specification)
    # if preprocess_specification == 'STANDARD':
    #     pass
    # else:
    #     raise ValueError('Unrecognized preprocess specification')


def fit_model(dataframes, model_specification, problem_specification, start_params=None):
    if model_specification['strategy'] == 'AR':
        model_specification = {
            'start_params': start_params,
            **model_specification
        }

    return {
        'AR': fit_model_ar,
        'VAR': fit_model_var,
        'SARIMAX': fit_model_sarimax,
        'ORDINARY_LEAST_SQUARES': fit_model_ordinary_linear_regression,
        'LOGISTIC_REGRESSION': fit_model_logistic_regression,
        'RANDOM_FOREST': fit_model_random_forest,
        'SUPPORT_VECTOR_CLASSIFIER': fit_model_svc
    }[model_specification['strategy']](dataframes, model_specification, problem_specification)


def fit_model_ar(dataframes, model_specification, problem_specification):
    """
    Return a fitted autoregression model

    @param dataframes:
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """

    dataframe = pd.concat(
        [i for i in [dataframes['targets'], dataframes['predictors']] if i is not None],
        axis=1)

    all = problem_specification['targets'] + problem_specification['predictors']
    date = next(iter(problem_specification.get('time', [])), None)
    exog = [i for i in problem_specification.get('exogenous', []) if i in all]
    endog = [i for i in dataframe.columns.values if i not in exog and i != date]

    if date is None:
        problem_specification['time'] = dataframe.index.name

    freq = get_freq(
        granularity_specification=problem_specification.get('timeGranularity'),
        series=dataframe.index)

    # UPDATE: statsmodels==0.10.x
    from statsmodels.tsa.ar_model import AR
    model = AR(
        endog=dataframe[endog],
        dates=dataframe.index,
        freq=freq)
    return model.fit(**filter_args(model_specification, ['start_params', 'maxlags', 'ic', 'trend']))

    # UPDATE: statsmodels==0.11.x
    # from statsmodels.tsa.ar_model import AutoReg
    # model = AutoReg(**{
    #     'endog': dataframe[endog],
    #     'exog': dataframe[exog] if exog else None,
    #     **filter_args(model_spec, ['lags', 'trend', 'seasonal', 'hold_back', 'period'])
    # })
    # return model.fit()


def fit_model_var(dataframes, model_specification, problem_specification):
    """
    Return a fitted vector autoregression model

    @param dataframes:
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """

    dataframe = pd.concat(
        [i for i in [dataframes['targets'], dataframes['predictors']] if i is not None],
        axis=1)

    all = problem_specification['targets'] + problem_specification['predictors']
    date = next(iter(problem_specification.get('time', [])), None)
    exog = [i for i in problem_specification.get('exogenous', []) if i in all]
    endog = [i for i in dataframe.columns.values if i not in exog and i != date]

    if date is None:
        problem_specification['time'] = dataframe.index.name

    freq = get_freq(
        granularity_specification=problem_specification.get('timeGranularity'),
        series=dataframe.index)

    from statsmodels.tsa.vector_ar.var_model import VAR

    model = VAR(
        endog=dataframe[endog],
        exog=dataframe[exog] if exog else None,
        dates=dataframe.index,
        freq=freq)
    # VAR cannot be trained with start_params, while AR can
    return model.fit(**filter_args(model_specification, ['maxlags', 'ic', 'trend']))


def fit_model_sarimax(dataframes, model_specification, problem_specification):
    """
    Return a fitted autoregression model

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """

    dataframe = pd.concat(
        [i for i in [dataframes['targets'], dataframes['predictors']] if i is not None],
        axis=1)
    all = problem_specification['targets'] + problem_specification['predictors']
    date = next(iter(problem_specification.get('time', [])), None)
    exog = [i for i in problem_specification.get('exogenous', []) if i in all]
    endog = [i for i in dataframe.columns.values if i not in exog and i != date]

    if date is None:
        problem_specification['time'] = dataframe.index.name

    freq = get_freq(
        granularity_specification=problem_specification.get('timeGranularity'),
        series=dataframe.index)

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        endog=dataframe[endog],
        exog=dataframe[exog] if exog else None,
        dates=dataframe.index,
        freq=freq,
        **filter_args(model_specification, [
            "order", "seasonal_order", "trend", "measurement_error",
            "time_varying_regression", "mle_regression", "simple_differencing",
            "enforce_stationarity", "enforce_invertibility", "hamilton_representation",
            "concentrate_scale", "trend_offset", "use_exact_diffuse"]))

    return model.fit(**filter_args(model_specification, [
        "start_params", "transformed", "includes_fixed", "cov_type", "cov_kwds",
        "method", "maxiter", "full_output", "disp", "callback", "return_params",
        "optim_score", "optim_complex_step", "optim_hessian", "flags", "low_memory"]))


def fit_model_ordinary_linear_regression(dataframes, model_specification, problem_specification):
    """
    Return a fitted linear regression model

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """

    from sklearn.linear_model import LinearRegression

    model = LinearRegression(
        **filter_args(model_specification, [
            "penalty", "dual", "tol", "C", "fit_intercept", "intercept_scaling",
            "class_weight", "random_state", "solver", "max_iter", "multi_class",
            "verbose", "warm_start", "n_jobs", "l1_ratio"]))

    model.fit(
        X=dataframes['predictors'],
        y=dataframes['targets'][problem_specification['targets'][0]],
        sample_weight=dataframes.get('weight')
    )

    return model


def fit_model_logistic_regression(dataframes, model_specification, problem_specification):
    """
    Return a fitted logistic regression model

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()

    model.fit(
        X=dataframes['predictors'],
        y=dataframes['targets'][problem_specification['targets'][0]],
        sample_weight=dataframes.get('weight'))

    return model


def fit_model_random_forest(dataframes, model_specification, problem_specification):
    """
    Return a random forest model

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        **filter_args(model_specification, [
            'bootstrap', 'class_weight', 'criterion', 'max_depth',
            'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
            'min_impurity_split', 'min_samples_leaf', 'min_samples_split',
            'min_weight_fraction_leaf', 'n_estimators'])
    )

    model.fit(
        X=dataframes['predictors'],
        y=dataframes['targets'][problem_specification['targets'][0]],
        sample_weight=dataframes.get('weight'))

    return model


def fit_model_svc(dataframes, model_specification, problem_specification):
    """
    Return a support vector classification model

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """
    from sklearn.svm import SVC

    model = SVC(
        **filter_args(model_specification, [
            'C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking',
            'probability', 'tol', 'cache_size', 'class_weight',
            'max_iter', 'decision_function_shape', 'break_ties', 'random_state'])
    )

    model.fit(
        X=dataframes['predictors'],
        y=dataframes['targets'][problem_specification['targets'][0]],
        sample_weight=dataframes.get('weight'))

    return model
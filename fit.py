from .utilities import Dataset, preprocess

from .utilities import (
    filter_args,
    get_freq,
    format_dataframe_time_index,
    split_time_series
)
import pandas as pd


# given a pipeline json and data, return a solution
def fit_pipeline(pipeline_specification, train_specification):
    # 1. load data
    dataframe = Dataset(train_specification['input']).get_dataframe()
    problem_specification = train_specification['problem']

    weights = problem_specification.get('weights')
    if weights and weights[0] in problem_specification['predictors']:
        problem_specification['predictors'].remove(weights[0])

    times = problem_specification.get('time')
    if times and times[0] in problem_specification['predictors']:
        problem_specification['predictors'].remove(times[0])

    if problem_specification['taskType'] == 'FORECASTING':
        # returns a dict of dataframes, one for each treatment, one observation per time unit
        dataframe_split = split_time_series(
            dataframe=dataframe,
            cross_section_names=problem_specification.get('crossSection', []))

        time = next(iter(problem_specification.get('time', [])), None)

        # targets cannot be exogenous, subset exogenous labels to the predictor set
        exog_names = [i for i in problem_specification.get('exogenous', []) if
                      i in problem_specification['predictors'] and
                      i not in problem_specification.get('crossSection', [])]
        # print('exog_names', exog_names)
        # target variables are not transformed, all other variables are transformed
        endog_non_target_names = [i for i in problem_specification['predictors'] if
                                  i not in exog_names and i != time and
                                  i not in problem_specification.get('crossSection', [])]
        endog_target_names = [i for i in problem_specification['targets'] if i != time]

        dataframes = {}
        preprocessors = {}

        for treatment_name in dataframe_split:
            treatment_data = format_dataframe_time_index(
                dataframe_split[treatment_name],
                date=time)
            if exog_names:
                exog, preprocess_exog = preprocess(
                    treatment_data[exog_names],
                    # pipeline_specification['preprocess'],
                    train_specification,
                    X=exog_names)
            else:
                exog, preprocess_exog = None, None

            if endog_non_target_names:
                endog_non_target, preprocess_endog = preprocess(
                    treatment_data[endog_non_target_names],
                    # pipeline_specification['preprocess'],
                    train_specification,
                    X=endog_non_target_names)

                # print(treatment_name)
                # print(endog_non_target_names)
                endog = pd.concat([
                    treatment_data[endog_target_names],
                    pd.DataFrame(endog_non_target, index=treatment_data.index)],
                    axis=1)
            else:
                endog, preprocess_endog = treatment_data[endog_target_names], None

            dataframes[treatment_name] = {
                'exogenous': exog,
                'endogenous': endog,
                'time': treatment_data.index,
                'weight': treatment_data[weights[0]] if weights else None
            }
            preprocessors[treatment_name] = {
                'exogenous': preprocess_exog,
                'endogenous': preprocess_endog
            }
    else:
        stimulus, preprocessor = fit_preprocess(
            dataframe[problem_specification['predictors']],
            pipeline_specification['preprocess'],
            train_specification)

        dataframes = {
            'targets': dataframe[problem_specification['targets']] if problem_specification['targets'] else None,
            'predictors': stimulus if problem_specification['predictors'] else None,
        }
        preprocessors = {
            'predictors': preprocessor
        }

        dataframes['weight'] = dataframe[weights[0]] if weights else None

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
            preprocessors=preprocessors)

    if model_specification['strategy'] in [
        'ORDINARY_LEAST_SQUARES', 'LOGISTIC_REGRESSION', 'RANDOM_FOREST', 'SUPPORT_VECTOR_CLASSIFIER'
    ]:
        return SciKitLearnWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            model=model,
            preprocessors=preprocessors)


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

    time = next(iter(problem_specification.get('time', [])), None)

    models = {}

    for treatment_name in dataframes:
        treatment_data = dataframes[treatment_name]
        if time is None:
            problem_specification['time'] = treatment_data['time'].name

        freq = get_freq(
            granularity_specification=problem_specification.get('timeGranularity'),
            series=dataframes['time'])

        # UPDATE: statsmodels==0.10.x
        from statsmodels.tsa.ar_model import AR
        model = AR(
            endog=treatment_data['endogenous'].astype(float),
            dates=treatment_data['time'],
            freq=freq)
        models[treatment_name] = model.fit(
            trend='nc',
            **filter_args(model_specification, ['start_params', 'maxlags', 'ic', 'trend']))

        # UPDATE: statsmodels==0.11.x
        # from statsmodels.tsa.ar_model import AutoReg
        # model = AutoReg(**{
        #     'endog': dataframe[endog],
        #     'exog': dataframe[exog] if exog else None,
        #     **filter_args(model_spec, ['lags', 'trend', 'seasonal', 'hold_back', 'period'])
        # })
        # return model.fit()

    return models


def fit_model_var(dataframes, model_specification, problem_specification):
    """
    Return a fitted vector autoregression model

    @param dataframes:
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """

    time = next(iter(problem_specification.get('time', [])), None)

    models = {}

    model_specification['drops'] = {}

    for treatment_name in dataframes:
        try:
            treatment_data = dataframes[treatment_name]
            if time is None:
                problem_specification['time'] = treatment_data['time'].name

            freq = get_freq(
                granularity_specification=problem_specification.get('timeGranularity'),
                series=treatment_data['time'])

            from statsmodels.tsa.vector_ar.var_model import VAR

            # endog_mask = treatment_data['endogenous'].T.duplicated()
            # print('xx')
            # print(treatment_data['endogenous'])
            endog_mask = treatment_data['endogenous'].var(axis=0) > 0
            endog = treatment_data['endogenous'][endog_mask.index[endog_mask]].astype(float)
            # print(endog.var(axis=0))
            # model_specification['drops'][treatment_name] = {'endogenous': endog_mask.tolist()}
            model_arguments = {'endog': endog}

            if treatment_data.get('exogenous'):
                # exog_mask = treatment_data['exogenous'].T.duplicated()
                exog_mask = treatment_data['exogenous'].var(axis=0) > 0
                # model_specification['drops'][treatment_name] = exog_mask.tolist()
                model_arguments['exog'] = treatment_data['exogenous'][exog_mask.index[exog_mask]].astype(float)

            model = VAR(
                **model_arguments,
                dates=treatment_data['time'],
                freq=freq)
            # VAR cannot be trained with start_params, while AR can
            models[treatment_name] = model.fit(
                **filter_args(model_specification, ['maxlags', 'ic', 'trend']))
        except Exception:
            # import traceback
            # traceback.print_exc()
            print('skipping cross section: ' + str(treatment_name))
    return models


def fit_model_sarimax(dataframes, model_specification, problem_specification):
    """
    Return a fitted autoregression model

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """

    time = next(iter(problem_specification.get('time', [])), None)

    models = {}

    for treatment_name, treatment_data in dataframes:
        treatment_data = dataframes[treatment_name]
        if time is None:
            problem_specification['time'] = treatment_data['time'].name

        freq = get_freq(
            granularity_specification=problem_specification.get('timeGranularity'),
            series=treatment_data['time'])

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        exog = treatment_data.get('exogenous')
        if exog is not None:
            exog = exog.astype(float)

        model = SARIMAX(
            endog=treatment_data['endogenous'].astype(float),
            exog=exog,
            dates=treatment_data['time'],
            freq=freq,
            **filter_args(model_specification, [
                "order", "seasonal_order", "trend", "measurement_error",
                "time_varying_regression", "mle_regression", "simple_differencing",
                "enforce_stationarity", "enforce_invertibility", "hamilton_representation",
                "concentrate_scale", "trend_offset", "use_exact_diffuse"]))

        models[treatment_name] = model.fit(**filter_args(model_specification, [
            "start_params", "transformed", "includes_fixed", "cov_type", "cov_kwds",
            "method", "maxiter", "full_output", "disp", "callback", "return_params",
            "optim_score", "optim_complex_step", "optim_hessian", "flags", "low_memory"]))

    return models


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
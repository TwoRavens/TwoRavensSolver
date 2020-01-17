from .utilities import Dataset, preprocess

from .utilities import (
    filter_args,
    format_dataframe_time_index,
    get_freq
)


# given a pipeline json and data, return a solution
def fit_pipeline(pipeline_specification, train_specification):
    # 1. load data
    dataset = Dataset(train_specification['input'])
    dataframe = dataset.get_dataframe()

    problem_specification = train_specification['problem']

    # 2. preprocess
    preprocessor = None
    if pipeline_specification.get('preprocess'):
        dataframe, preprocessor = fit_preprocess(
            dataframe,
            pipeline_specification['preprocess'],
            problem_specification)

    # 3. modeling
    model_specification = pipeline_specification['model']
    model = fit_model(dataframe, model_specification, problem_specification)

    # 4. wrap and save
    if model_specification['strategy'] in ['ar']:
        from .model import StatsModelsWrapper
        return StatsModelsWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            model=model,
            preprocess=preprocessor
        )


def fit_preprocess(dataframe, preprocess_specification, train_specification):
    # TODO: more varied preprocessing based on preprocess specification
    if preprocess_specification == 'STANDARD':
        return preprocess(dataframe, train_specification)
    else:
        raise ValueError('Unrecognized preprocess specification')


def fit_model(dataframe, model_specification, problem_specification, start_params=None):
    if model_specification['strategy'] == 'ar':
        model_specification = {
            'start_params': start_params,
            **model_specification
        }

    return {
        'ar': fit_model_ar,
    }[model_specification['strategy']](dataframe, model_specification, problem_specification)


def fit_model_ar(dataframe, model_specification, problem_specification):
    """
    Return a fitted autoregression model

    @param dataframe:
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """
    exog = problem_specification['predictors']
    endog = problem_specification['targets']
    date = problem_specification.get('time')

    dataframe, log = format_dataframe_time_index(dataframe, date)

    if date is None:
        problem_specification['time'] = dataframe.index.name

    freq = get_freq(problem_specification.get('timeGranularity'), dataframe.index)

    if len(endog) == 1:
        # UPDATE: statsmodels==0.10.x
        from statsmodels.tsa.ar_model import AR
        model = AR(
            endog=dataframe[endog],
            dates=dataframe.index,
            freq=freq
        )
        return model.fit(**filter_args(model_specification, ['start_params', 'maxlags', 'ic', 'trend']))

        # UPDATE: statsmodels==0.11.x
        # from statsmodels.tsa.ar_model import AutoReg
        # model = AutoReg(**{
        #     'endog': dataframe[endog],
        #     'exog': dataframe[exog] if exog else None,
        #     **filter_args(model_spec, ['lags', 'trend', 'seasonal', 'hold_back', 'period'])
        # })
        # return model.fit()

    elif len(endog) > 1:
        from statsmodels.tsa.vector_ar.var_model import VAR
        print('freq', freq)
        print(type(freq))
        print(dataframe)
        model = VAR(
            endog=dataframe[endog],
            exog=dataframe[exog] if exog else None,
            dates=dataframe.index,
            freq=freq
        )
        # VAR cannot be trained with start_params, while AR can
        return model.fit(**filter_args(model_specification, ['maxlags', 'ic', 'trend']))

    raise ValueError('At least one endogenous variable is needed for autoregression')


from .utilities import Dataset, preprocess

from .utilities import (
    filter_args,
    get_freq,
    format_dataframe_order_index,
    split_time_series
)

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, \
    Lasso, LassoLars, ElasticNet, OrthogonalMatchingPursuit, RidgeClassifierCV, LogisticRegressionCV, \
    RidgeCV
from sklearn.ensemble import\
    RandomForestClassifier, RandomForestRegressor,\
    AdaBoostClassifier, AdaBoostRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB

import pandas as pd
import numpy as np
import inspect


_LOSS_FUNCTIONS = {
    'MEAN_SQUARED_ERROR': 'squared_loss',
    'R_SQUARED': 'rooted_squared_loss',
    'MEAN_ABSOLUTE_ERROR': 'mean_absolute_loss',
}


# given a pipeline json and data, return a solution
def fit_pipeline(pipeline_specification, train_specification):
    # 1. load data
    dataframe = Dataset(train_specification['input']).get_dataframe()
    problem_specification = train_specification['problem']

    # Add performanceMetric into train_specification
    loss_name = train_specification.get("performanceMetric", {'metric': "MEAN_SQUARED_ERROR"})
    problem_specification['performanceMetric'] = loss_name

    # Test code for dummy preprocessor return
    if 'is_temporal' not in problem_specification:
        problem_specification['is_temporal'] = False
    if 'data_format' not in problem_specification:
        problem_specification['date_format'] = None

    weights = problem_specification.get('weights')
    if weights and weights[0] in problem_specification['predictors']:
        problem_specification['predictors'].remove(weights[0])

    ordering_column = problem_specification.get('forecastingHorizon', {}).get('column')
    if ordering_column and ordering_column in problem_specification['predictors']:
        problem_specification['predictors'].remove(ordering_column)

    # drop null values in the target column
    dataframe = dataframe[dataframe[problem_specification['targets']].notnull().all(1)]

    if problem_specification['taskType'] == 'FORECASTING':
        dataframes, preprocessors = fit_forecast_preprocess(dataframe, problem_specification,
                                                            train_specification, weights)
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
    model_specification['probability'] = True  # Aux operation for Sklearn-SVC
    model = fit_model(dataframes, model_specification, problem_specification)

    # model_specification['library'] = 'statsmodels'

    # 4. wrap and save
    from .model import StatsModelsWrapper, SciKitLearnWrapper
    if model_specification['library'] == 'statsmodels':
        return StatsModelsWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            model=model,
            preprocessors=preprocessors)

    if model_specification['library'] == 'sklearn':
        return SciKitLearnWrapper(
            pipeline_specification=pipeline_specification,
            problem_specification=problem_specification,
            model=model,
            preprocessors=preprocessors)

    # if model_specification['library'] == 'torch':
    #     return TorchModelsWrapper(
    #         pipeline_specification=pipeline_specification,
    #         problem_specification=problem_specification,
    #         model=model,
    #         preprocessors=preprocessors)


def fit_preprocess(dataframe, preprocess_specification, train_specification):
    # TODO: more varied preprocessing based on preprocess specification
    return preprocess(dataframe, train_specification)
    # if preprocess_specification == 'STANDARD':
    #     pass
    # else:
    #     raise ValueError('Unrecognized preprocess specification')


def fit_forecast_preprocess(dataframe, problem_specification, train_specification, weights):
    # Move the inner block here, so it can be used by ta2_interface
    # returns a dict of dataframes, one for each treatment, one observation per time unit
    dataframe_split = split_time_series(
        dataframe=dataframe,
        cross_section_names=problem_specification.get('crossSection', []))

    ordering_column = problem_specification.get("forecastingHorizon", {}).get('column')

    # targets cannot be exogenous, subset exogenous labels to the predictor set
    exog_names = [i for i in problem_specification.get('exogenous', []) if
                  i in problem_specification['predictors'] and
                  i not in problem_specification.get('crossSection', [])]
    # print('exog_names', exog_names)
    # target variables are not transformed, all other variables are transformed
    endog_non_target_names = [i for i in problem_specification['predictors'] if
                              i not in exog_names and i != ordering_column and
                              i not in problem_specification.get('crossSection', [])]
    endog_target_names = [i for i in problem_specification['targets'] if i != ordering_column]

    dataframes = {}
    preprocessors = {}

    for treatment_name in dataframe_split:
        treatment_data, mapping_dic = format_dataframe_order_index(
            dataframe_split[treatment_name],
            is_date=problem_specification['is_temporal'],
            date_format=problem_specification['date_format'].get(ordering_column),
            order_column=ordering_column
        )
        if exog_names:
            exog, preprocess_exog = preprocess(
                treatment_data[exog_names],
                train_specification,
                X=exog_names)
        else:
            exog, preprocess_exog = None, None

        if endog_non_target_names:
            endog_non_target, preprocess_endog = preprocess(
                treatment_data[endog_non_target_names],
                train_specification,
                X=endog_non_target_names)

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
            'weight': treatment_data[weights[0]] if weights else None,
            'tgt_names': endog_target_names,
            'index_mapping': mapping_dic
        }
        preprocessors[treatment_name] = {
            'exogenous': preprocess_exog,
            'endogenous': preprocess_endog
        }

    return dataframes, preprocessors


def fit_model(dataframes, model_specification, problem_specification, start_params=None):
    if model_specification['strategy'] == 'AR':
        model_specification = {
            'start_params': start_params,
            **model_specification
        }

    temp_key = model_specification['strategy']
    if temp_key.startswith('SARIMAX'):
        temp_key = 'SARIMAX'
        
    return {
        'AR': fit_model_ar,
        'AR_NN': fit_model_ar_ann,
        'VAR_NN': fit_model_var_ann,
        'VAR': fit_model_var,
        'SARIMAX': fit_model_sarimax,
        'ORDINARY_LEAST_SQUARES': factory_fit_model_sklearn(LinearRegression),
        'LOGISTIC_REGRESSION': factory_fit_model_sklearn(LogisticRegression),
        'LOGISTIC_REGRESSION_CV': factory_fit_model_sklearn(LogisticRegressionCV),
        'RANDOM_FOREST': factory_fit_model_sklearn(RandomForestClassifier),
        'SUPPORT_VECTOR_CLASSIFIER': factory_fit_model_sklearn(SVC),
        "RIDGE_CLASSIFIER": factory_fit_model_sklearn(RidgeClassifier),
        "RIDGE_CLASSIFIER_CV": factory_fit_model_sklearn(RidgeClassifierCV),
        "RIDGE_CV": factory_fit_model_sklearn(RidgeCV),
        "RANDOM_FOREST_REGRESSOR": factory_fit_model_sklearn(RandomForestRegressor),
        "SUPPORT_VECTOR_REGRESSION": factory_fit_model_sklearn(SVR),
        "K_NEIGHBORS_CLASSIFIER": factory_fit_model_sklearn(KNeighborsClassifier),
        "K_NEIGHBORS_REGRESSOR": factory_fit_model_sklearn(KNeighborsRegressor),
        "DECISION_TREE_CLASSIFIER": factory_fit_model_sklearn(DecisionTreeClassifier),
        "DECISION_TREE_REGRESSOR": factory_fit_model_sklearn(DecisionTreeRegressor),
        "LASSO_REGRESSION": factory_fit_model_sklearn(Lasso),
        "LASSO_REGRESSION_LARS": factory_fit_model_sklearn(LassoLars),
        "ELASTIC_NET": factory_fit_model_sklearn(ElasticNet),
        "ORTHOGONAL_MATCHING": factory_fit_model_sklearn(OrthogonalMatchingPursuit),
        "ADABOOST_CLASSIFIER": factory_fit_model_sklearn(AdaBoostClassifier),
        "ADABOOST_REGRESSOR": factory_fit_model_sklearn(AdaBoostRegressor),
        "GRADIENT_BOOSTING_CLASSIFIER": factory_fit_model_sklearn(GradientBoostingClassifier),
        "GRADIENT_BOOSTING_REGRESSOR": factory_fit_model_sklearn(GradientBoostingRegressor),
        "LINEAR_DISCRIMINANT_ANALYSIS": factory_fit_model_sklearn(LinearDiscriminantAnalysis),
        "QUADRATIC_DISCRIMINANT_ANALYSIS": factory_fit_model_sklearn(QuadraticDiscriminantAnalysis),
        "GAUSSIAN_PROCESS_CLASSIFIER": factory_fit_model_sklearn(GaussianProcessClassifier),
        "GAUSSIAN_PROCESS_REGRESSOR": factory_fit_model_sklearn(GaussianProcessRegressor),
        "MULTINOMIAL_NAIVE_BAYES": factory_fit_model_sklearn(MultinomialNB),
        "GAUSSIAN_NAIVE_BAYES": factory_fit_model_sklearn(GaussianNB),
        "COMPLEMENT_NAIVE_BAYES": factory_fit_model_sklearn(ComplementNB),
        "TRA_AVERAGE": factory_fit_traditional("AVERAGE"),
        "TRA_NAIVE": factory_fit_traditional("NAIVE"),
        "TRA_DRIFT": factory_fit_traditional("DRIFT"),
    }[temp_key](dataframes, model_specification, problem_specification)


def fit_model_ar(dataframes, model_specification, problem_specification):
    """
    Return a fitted autoregression model

    @param dataframes:
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """

    ordering_column = problem_specification.get("forecastingHorizon", {}).get('column')

    models = {}

    for treatment_name in dataframes:
        try:
            treatment_data = dataframes[treatment_name]
            # print(treatment_data)
            if ordering_column is None:
                problem_specification.setDefault('forecastingHorizon', {})
                problem_specification['forecastingHorizon']['column'] = treatment_data['time'].name

            # freq = get_freq(
            #     granularity_specification=problem_specification.get('timeGranularity'),
            #     series=treatment_data['time'])

            # UPDATE: statsmodels==0.10.x
            from statsmodels.tsa.ar_model import AR
            model = AR(
                endog=treatment_data['endogenous'].astype(float),
                dates=treatment_data['time'],
                # freq=treatment_data['endogenous'].index.freq
                # freq=freq
            )
            models[treatment_name] = model.fit(
                **filter_args(model_specification, ['start_params', 'maxlags', 'ic', 'trend']))

            # UPDATE: statsmodels==0.11.x
            # from statsmodels.tsa.ar_model import AutoReg
            # model = AutoReg(**{
            #     'endog': dataframe[endog],
            #     'exog': dataframe[exog] if exog else None,
            #     **filter_args(model_spec, ['lags', 'trend', 'seasonal', 'hold_back', 'period'])
            # })
            # return model.fit()
        except Exception:
            # import traceback
            # traceback.print_exc()
            print('skipping cross section: ' + str(treatment_name))

    return models


def fit_model_var(dataframes, model_specification, problem_specification):
    """
    Return a fitted vector autoregression model

    @param dataframes:
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """

    ordering_column = problem_specification.get("forecastingHorizon", {}).get('column')

    models = {}

    model_specification['drops'] = {}

    for treatment_name in dataframes:
        try:
            treatment_data = dataframes[treatment_name]
            if ordering_column is None:
                problem_specification.setDefault('forecastingHorizon', {})
                problem_specification['forecastingHorizon']['column'] = treatment_data['time'].name

            freq = get_freq(
                granularity_specification=problem_specification.get('timeGranularity'),
                series=treatment_data['time'])

            from statsmodels.tsa.vector_ar.var_model import VAR

            # endog_mask = treatment_data['endogenous'].T.duplicated()
            # print('xx')
            # print(treatment_data['endogenous'])
            endog_mask = treatment_data['endogenous'].var(axis=0) > 0
            # print(endog_mask)
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

    ordering_column = problem_specification.get("forecastingHorizon", {}).get('column')

    models = {}

    for treatment_name in dataframes:
        treatment_data = dataframes[treatment_name]
        if ordering_column is None:
            problem_specification.setDefault('forecastingHorizon', {})
            problem_specification['forecastingHorizon']['column'] = treatment_data['time'].name

        # freq = get_freq(
        #     granularity_specification=problem_specification.get('timeGranularity'),
        #     series=treatment_data['time'])
        # print(freq)
        # print(treatment_data['endogenous'])

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        exog = treatment_data.get('exogenous')
        if exog is not None:
            exog = exog.astype(float)

        model = SARIMAX(
            endog=treatment_data['endogenous'].astype(float),
            exog=exog,
            dates=treatment_data['time'],
            # freq=freq,
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


def factory_fit_model_sklearn(sklearn_class):
    """
    Return a function that will fit the provided class

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """
    def fit_model(dataframes, model_specification, problem_specification):
        """
        Return a fitted model

        @param dataframes:
        @param model_specification:
        @param problem_specification:
        """
        model = sklearn_class(
            **filter_args(
                model_specification,
                list(inspect.signature(sklearn_class.__init__).parameters.keys())))

        model.fit(
            **filter_args(
                {
                    'X': dataframes['predictors'],
                    'y': dataframes['targets'][problem_specification['targets'][0]],
                    'sample_weight': dataframes.get('weight')
                },
                list(inspect.signature(sklearn_class.fit).parameters.keys())))

        if hasattr(model, 'classes_'):
            # This is a classification problem, store the label for future use
            tmp_list = model.classes_.tolist()
            problem_specification['clf_classes'] = [str(item) for item in tmp_list]
        return model
    return fit_model


def fit_model_ar_ann(dataframes, model_specification, problem_specification):
    """
    Return a fitted autoregression neural network model
    @param dataframes: ordered by time index
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """
    # 'Y' variable is in the first column, AR only requires 'Y' value
    ordering_column = problem_specification.get("forecastingHorizon", {}).get('column')
    back_steps = model_specification.get('back_steps', 1)  # At least 1 time step is required
    loss_func = problem_specification.get('performanceMetric').get('metric')
    loss_func = 'MEAN_SQUARED_ERROR' if (not loss_func or loss_func not in _LOSS_FUNCTIONS) else loss_func
    models = dict()

    for treatment_name in dataframes:
        treatment_data = dataframes[treatment_name]
        if ordering_column is None:
            problem_specification.setDefault('forecastingHorizon', {})
            problem_specification['forecastingHorizon']['column'] = treatment_data['time'].name

        # Only considering endogenous features now
        if treatment_data['exogenous']:
            print('Exogenous features will not be considered now.')

        container = treatment_data['endogenous'].astype(float)
        tgt_name = container.columns[0]
        y_column = container[tgt_name]
        tmp_block = container.drop(columns=[tgt_name])
        history_points = y_column.tail(back_steps)

        tgt_y, tgt_x = y_column, tmp_block

        # Build training matrix, can be moved to a new auxiliary function
        for step in range(1, back_steps + 1):
            tgt_y = tgt_y.drop(tgt_y.index[0])
            tmp_x = container.shift(step)
            tmp_x.columns = ['{}_minus_{}'.format(col, step) for col in tmp_x.columns]
            tgt_x = pd.concat((tgt_x, tmp_x), axis=1)

        train_x, train_y = tgt_x.dropna(), tgt_y.dropna()

        from .nn_models.NlayerMLP import ModMLPForecaster

        # Training model for current df
        model = ModMLPForecaster(loss=_LOSS_FUNCTIONS[loss_func])
        model.fit(train_x, train_y)

        # history points should be stored for future inference
        model.set_history(history_points, treatment_data['time'])

        models[treatment_name] = model

    return models


def fit_model_var_ann(dataframes, model_specification, problem_specification):
    """
    Return a fitted autoregression model
    @param dataframes:
    @param model_specification: {'lags': int, ...}
    @param problem_specification:
    """
    # Assume the dataframes is already in order
    # 'Y' variable is in the first column, AR only requires 'Y' value
    ordering_column = problem_specification.get("forecastingHorizon", {}).get('column')
    back_steps = model_specification.get('back_steps', 1)  # At least 1 time step is required
    loss_func = problem_specification.get('performanceMetric').get('metric')
    loss_func = 'MEAN_SQUARED_ERROR' if (not loss_func or loss_func not in _LOSS_FUNCTIONS) else loss_func

    models = dict()

    for treatment_name in dataframes:
        treatment_data = dataframes[treatment_name]
        if ordering_column is None:
            problem_specification.setDefault('forecastingHorizon', {})
            problem_specification['forecastingHorizon']['column'] = treatment_data['time'].name

        # Only considering endogenous features now
        if treatment_data['exogenous']:
            print('Exogenous features will not be considered now.')

        # container = treatment_data['endogenous'].astype(float)
        endog_mask = treatment_data['endogenous'].var(axis=0) > 0
        container = treatment_data['endogenous'][endog_mask.index[endog_mask]].astype(float)
        y_column = container  # Predict Y, X1, X2 ... simultaneously
        tmp_block = pd.DataFrame()
        history_points = y_column.tail(back_steps)

        tgt_y, tgt_x = y_column, tmp_block

        for step in range(1, back_steps + 1):
            tgt_y = tgt_y.drop(tgt_y.index[0])
            tmp_x = container.shift(step)
            tmp_x.columns = ['{}_minus_{}'.format(col, step) for col in tmp_x.columns]
            tgt_x = pd.concat((tgt_x, tmp_x), axis=1)

        train_x, train_y = tgt_x.dropna(), tgt_y.dropna()

        from .nn_models.NlayerMLP import ModMLPForecaster

        # Training model for current df
        model = ModMLPForecaster(loss=_LOSS_FUNCTIONS[loss_func], num_tgt=len(treatment_data['tgt_names']))
        model.fit(train_x, train_y)

        model.set_history(history_points, treatment_data['time'])

        models[treatment_name] = model

    return models


class DummyTra(object):
    # Dummy class that hold the information of traditional methods
    def __init__(self, method=None, value=None, num_tgt=1):
        self.method, self.value = method, value
        # Index object that should be pd.DateTimeIndex or ...
        self._index = None
        self.num_tgt = num_tgt

    def __str__(self):
        return self.value
    pass


def factory_fit_traditional(method_name):
    """
    Return a function that will fit the provided class

    @param dataframes:
    @param model_specification:
    @param problem_specification:
    """

    def fit_model(dataframes, model_specification, problem_specification):
        """
        Return a fitted model -- Seems only an scalar will do the job

        @param dataframes:
        @param model_specification:
        @param problem_specification:
        """
        ordering_column = problem_specification.get("forecastingHorizon", {}).get('column')

        models = dict()

        for treatment_name in dataframes:
            treatment_data = dataframes[treatment_name]
            if ordering_column is None:
                problem_specification.setDefault('forecastingHorizon', {})
                problem_specification['forecastingHorizon']['column'] = treatment_data['time'].name

            # Only considering endogenous features now
            if treatment_data['exogenous']:
                print('Exogenous features will not be considered now.')

            container = treatment_data['endogenous'][treatment_data['tgt_names']].astype(float)

            # Traditional method won't use any non-tgt features
            model = DummyTra(method_name, num_tgt=len(treatment_data['tgt_names']))
            model._index = treatment_data['time']

            if "AVERAGE" == method_name:
                # Naive approach, stores the average of observations
                tmp = container.to_numpy()
                model.value = np.mean(tmp, axis=0).reshape(1, -1)
            elif "NAIVE" == method_name:
                # Naive approach, get last history
                model.value = container.tail(1).to_numpy().reshape(1, -1)
            elif "DRIFT" == method_name:
                t1, t2 = container.head(1).to_numpy(), container.tail(1).to_numpy()
                gap = len(container.index)
                model.value = {'slope': (t2 - t1) / float(gap), 'pivot': t2}

            models[treatment_name] = model

        return models
    return fit_model

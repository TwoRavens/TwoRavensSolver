import tworaven_solver


pipeline_specification = {
    'preprocess': None,
    'model': {
        'strategy': 'ar'
    }
}


train_specification = {
    "problem": {
        "predictors": [],
        "targets": ['Sales'],
        "time": "Month"
    },
    "input": {
        "name": "in-sample",
        "resource_uri": "file://" + '/home/shoe/TwoRavens/dev_scripts/time_series_data/shampoo.csv'
    }
}


model_tworavens = tworaven_solver.fit_pipeline(
    pipeline_specification=pipeline_specification,
    train_specification=train_specification)

print(model_tworavens.forecast(3))

train_specification['problem']['targets'].append('Sales')

model_tworavens = tworaven_solver.fit_pipeline(
    pipeline_specification=pipeline_specification,
    train_specification=train_specification)

print(model_tworavens.forecast(3))

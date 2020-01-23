
strategies = {
    'FORECASTING': {
        'UNIVARIATE': ['AR', 'SARIMAX'],
        'MULTIVARIATE': ['VAR']
    },
    'CLASSIFICATION': {
        'BINARY': [
            'LOGISTIC_REGRESSION'
        ],
        'MULTICLASS': [
            'RANDOM_FOREST'
        ],
        'MULTILABEL': [
            'RANDOM_FOREST'
        ]
    },
    'REGRESSION': {
        'UNIVARIATE': [
            'ORDINARY_LEAST_SQUARES'
        ]
    }
}


class SearchManager(object):
    def __init__(self, system_params, problem_specification):
        self.problem_specification = problem_specification
        self.system_params = system_params

        print(problem_specification)
        task = problem_specification['taskType']
        subtask = problem_specification.get('taskSubtype')

        # TODO: forecasting subtypes need rework
        if problem_specification['taskType'] == 'FORECASTING':
            variables = problem_specification['targets'] + problem_specification['predictors']
            subtask = 'MULTIVARIATE' if len(variables) > 2 else 'UNIVARIATE'

        print(subtask)
        self.generator = iter(strategies.get(task, {}).get(subtask, []))

    def get_pipeline_specification(self):
        try:
            strategy = next(self.generator)
        except StopIteration:
            return

        return {
            'preprocess': None,
            'model': {
                'strategy': strategy,
                **self.system_params.get(strategy, {})
            }
        }

    def metalearn_result(self, pipeline_specification, scores):
        pass

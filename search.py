
strategies = {
    'FORECASTING': [
        'ar',
        # 'arima'
    ],
}


class SearchManager(object):
    def __init__(self, task, subtask, system_params):
        self.task = task
        self.subtask = subtask
        self.system_params = system_params

        self.generator = iter(strategies.get(self.task, []))

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

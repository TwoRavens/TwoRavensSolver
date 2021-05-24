import pytest

from tworaven_solver.tests.shared import problems

from tworaven_solver.search import find_pipeline
from tworaven_solver.solution import Solution
from tworaven_solver.utilities import TrainSpecification

import itertools

TS_MODELS = ['VAR', 'AR', 'PROPHET', 'AR_NN']
PROBLEM_NAMES = ['sunspots', 'appliance', 'shampoo']


@pytest.mark.parametrize("strategy,problem_name", itertools.product(TS_MODELS, PROBLEM_NAMES))
def test_any(strategy, problem_name, plot=False):
    train_spec = TrainSpecification(problems[problem_name]['train_specification'])

    solution = Solution(
        pipeline_specification=find_pipeline(strategy),
        train_specification=train_spec)
    solution.fit()

    fitted = solution.fitted_values()

    prediction = solution.predict(solution.get_future_dataframe(10))

    if plot:
        dataframe = train_spec.dataset('input').get_dataframe(usecols=train_spec.problem.train)
        solution.preprocessor.format_dataframe_date_index(dataframe)

        target = solution.problem.targets[0]

        # dataframe['fitted'] = fitted[target]
        # print(dataframe[[target, 'fitted']].head())

        import matplotlib.pyplot as plt
        plt.plot(target, data=dataframe.tail(20))
        plt.plot(target, data=fitted.tail(20))
        plt.plot(target, data=prediction)
        plt.show()


if __name__ == '__main__':
    # test_any('SARIMAX', 'sunspots', plot=True)
    test_any('AR', 'appliance', plot=True)
    # test_any('AR', 'sunspots', plot=True)

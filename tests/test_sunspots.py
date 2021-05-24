from tworaven_solver.tests.shared import problems

import tworaven_solver
from tworaven_solver.search import SearchManager
from tworaven_solver.utilities import TrainSpecification


def run_problem(specs):

    train_spec = TrainSpecification(specs['train_specification'])
    dataframe = train_spec.dataset('input').get_dataframe(usecols=train_spec.problem.train)

    search_manager = SearchManager(train_spec.problem)

    for pipeline_spec in list(search_manager)[:1]:
        print('pipeline spec', pipeline_spec)
        model = tworaven_solver.model.Solution.fit(pipeline_spec, train_spec)

        res = model.fitted_values()
        print('fitted values')
        print(res)

        res = model.predict(dataframe)
        print(res)


# end = model.model.model._index[-1]
# start = model.model.model._index[0]
#
# model.model.plot_forecast()
#
# import matplotlib.pyplot as plt
#
# plt.show()
#
# dataframe = pandas.read_csv(data_path)
# dataframe['Month'] = pandas.to_datetime(dataframe['Month'])
# # dataframe = dataframe.set_index('Month')
#
#
# def infer_freq(series):
#
#     def approx_seconds(offset):
#         offset = pd.tseries.frequencies.to_offset(offset)
#         try:
#             return offset.nanos / 1E9
#         except ValueError:
#             pass
#
#         date = datetime.now()
#         return ((offset.rollback(date) - offset.rollforward(date)) * offset.n).total_seconds()
#
#     # infer frequency from every three-pair of records
#     candidate_frequencies = set()
#     for i in range(len(series) - 3):
#         candidate_frequency = pd.infer_freq(series[i:i + 3])
#         if candidate_frequency:
#             candidate_frequencies.add(candidate_frequency)
#
#     # sort inferred frequency by approximate time durations
#     return sorted([(i, approx_seconds(i)) for i in candidate_frequencies], key=lambda x: x[1])[-1][0]
#
#
# freq = infer_freq(dataframe['Month'])
#
# dataframe = dataframe.set_index('Month')
# dataframe_temp = dataframe.resample(freq).mean()
# numeric_columns = list(dataframe.select_dtypes(include=[np.number]).columns.values)
# categorical_columns = [i for i in dataframe.columns.values if i not in numeric_columns]
#
# for dropped_column in categorical_columns:
#     dataframe_temp[dropped_column] = dataframe[dropped_column]
#
# print(numeric_columns)
# print(categorical_columns)
# dataframe = pd.DataFrame(ColumnTransformer(transformers=[
#     ('numeric', SimpleImputer(strategy='median'), numeric_columns),
#     ('categorical', SimpleImputer(strategy='most_frequent'), categorical_columns)
# ]).fit_transform(dataframe_temp), index=dataframe_temp.index, columns=dataframe_temp.columns)
#
# print(dataframe)
#
#
# # split_index = int(len(dataframe) * .75)
# # history = dataframe.head(split_index)
# # future = dataframe.head(-split_index)
# #
# # print(model_tworavens.forecast(history, 3))
# # print(future.head(3))
# # train_specification['problem']['targets'].append('Sales')
# #
# # model_tworavens = tworaven_solver.fit_pipeline(
# #     pipeline_specification=pipeline_specification,
# #     train_specification=train_specification)
# #
# # print(model_tworavens.forecast(3))

if __name__ == "__main__":
    run_problem(problems['sunspots'])
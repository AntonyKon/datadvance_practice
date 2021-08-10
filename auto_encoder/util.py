import category_encoders as ce
from .encoders import DummyEncoder, SubsampleEncoder
import numpy as np
import pandas as pd
from collections import defaultdict


class ConfigurationManager:
    """
    Class, which build configuration for auto_encoder.Builder.
    >>> manager = ConfigurationManager(x, y, encoders=[SubsampleEncoder, DummyEncoder, ce.BinaryEncoder, ce.OneHotEncoder, ce.LeaveOneOutEncoder])
    >>> config = manager.build_configuration(categorical_variables)
    """

    class metrics:

        def __init__(self, max_shape, min_elements):
            self.max_shape = max_shape
            self.min_elements = min_elements

        def std_limit(self, df):
            return max(0, df.std() - np.log2((df.max() - df.min()) + 1) * 225)

        def elements_limit(self, df):
            return np.count_nonzero(df.to_numpy() < self.min_elements) * 100

        def dimension_limit(self, dimension):
            return max(0, dimension - self.max_shape) * 100

        def data_losing(self, transformed_values, column_values):
            return (column_values.nunique() - transformed_values.nunique()) * 100

        def diff_limit(self, transformed_values):
            unique_elements = transformed_values.unique()
            unique_count = transformed_values.nunique()

            real_diff = np.median(np.abs(np.diff(unique_elements)))  # calculating median of real differences
            ideal_diff = (np.max(unique_elements) - np.min(unique_elements)) / (
                        unique_count - 1)  # arithmetic progression (calculating d)

            return max(0, real_diff - ideal_diff)

    def __init__(self, x, y, encoders=None):
        self.encoders = encoders
        self.x = pd.DataFrame(x)
        self.y = y

        if self.encoders is None:
            self.encoders = [SubsampleEncoder, DummyEncoder, ce.HashingEncoder, ce.OneHotEncoder, ce.LeaveOneOutEncoder]

    def build_configuration(self, categorical_variables):
        if len(categorical_variables) == 0:
            return []

        categorical_variables = self.x.iloc[:, categorical_variables].columns.to_numpy()
        actual_shape = self.x.shape[1]
        metric_checker = self.metrics(max_shape=2.5 * actual_shape, min_elements=2 * actual_shape + 4)

        configuration = defaultdict(list)

        for column in categorical_variables:
            scores = dict()

            for encoder in self.encoders:
                scores[encoder] = self.__validate(encoder, metric_checker, actual_shape, column)

            optimal_encoder = min(scores, key=scores.get)

            if isinstance(optimal_encoder(), SubsampleEncoder):
                configuration[ce.OrdinalEncoder].append(column)
            configuration[optimal_encoder].append(column)

            tmp_transformed = optimal_encoder(column).fit_transform(self.x, self.y)
            actual_shape += tmp_transformed.columns.size - self.x.columns.size

            dict_for_sort = defaultdict(int)
            dict_for_sort[SubsampleEncoder] = 2
            dict_for_sort[ce.OrdinalEncoder] = 1

        return sorted([encoder(cols=configuration[encoder]) for encoder in configuration],
                      key=lambda encoder: dict_for_sort[type(encoder)])

    def __validate(self, encoder, metric_checker, actual_shape, column):
        score = 0
        dataset_size = pd.Series([len(self.x)])

        if isinstance(encoder(), SubsampleEncoder):
            dataset_size = self.x.groupby(column).size()
            score += metric_checker.std_limit(dataset_size)

        score += metric_checker.elements_limit(dataset_size)
        tmp_transformed = encoder(cols=column).fit(self.x, self.y).transform(self.x) # leaveoneout turns into target encoder without y
        actual_shape += tmp_transformed.shape[1] - self.x.shape[1]

        score += metric_checker.dimension_limit(actual_shape)

        if column in tmp_transformed.columns:
            score += metric_checker.data_losing(tmp_transformed[column], self.x[column])
            score += metric_checker.diff_limit(tmp_transformed[column])

        return score

import category_encoders as ce
from .encoders import DummyEncoder, SubsampleEncoder
import numpy as np
import pandas as pd


def _chisquare(observed, expected=None):
    observed = pd.Series(observed).to_numpy(dtype=float)
    if expected is None:
        expected = np.full_like(observed, np.mean(observed), dtype=float)

    return np.sum(((observed - expected) ** 2) / expected)


class ConfigurationManager:
    """
    Class, which builds configuration for auto_encoder.Builder.
    >>> encoders = [SubsampleEncoder, DummyEncoder, ce.BinaryEncoder, ce.OneHotEncoder, ce.LeaveOneOutEncoder]
    >>> categorical_variables = [1, 2, 3]
    >>> manager = ConfigurationManager(x, y, encoders=encoders, technique='RSM')
    >>> config = manager.build_configuration(categorical_variables)
    """

    class metrics:

        def __init__(self, initial_dim, min_elements, max_elements=None):
            self.initial_dim = initial_dim
            self.min_elements = min_elements
            self.max_elements = max_elements

        def disribution_metric(self, subsample_sizes):  # 704.17172013
            dof = (len(subsample_sizes) - 1) or 1  # dimenstions of freedom
            return np.sqrt(_chisquare(subsample_sizes)) / dof  # .apply(lambda size: size / subsample_sizes.max())

        def elements_limit(self, subsample_sizes):  # 944.24572173
            mask = self.__elements_min_limit(subsample_sizes)

            if self.max_elements:
                mask += self.__elements_max_limit(subsample_sizes)

            return np.count_nonzero(mask)

        def sample_count_metric(self, subsamples):  # 1.96859905
            return len(subsamples) - 1

        def __elements_min_limit(self, subsample_sizes):
            return subsample_sizes.to_numpy() < self.min_elements

        def __elements_max_limit(self, subsample_sizes):
            return subsample_sizes.to_numpy() > self.max_elements

        def dimension_increase(self, encoded_dim):  # 134.96166354
            return encoded_dim / self.initial_dim - 1

        def data_losing(self, transformed_values, column_values):  # 539.48631667
            return abs(len(column_values) - len(transformed_values))

        def diff_metric(self, transformed_values):  # 29.11024482
            unique_elements = transformed_values.unique()
            unique_count = transformed_values.nunique()

            real_diff = np.abs(np.diff(unique_elements)).astype(float)  # calculating median of real differences
            dof = (len(real_diff) - 1) or 1
            ideal_diff = np.ptp(unique_elements) / (unique_count - 1)  # arithmetic progression (calculating d)

            return np.sqrt(_chisquare(real_diff, np.full_like(real_diff, ideal_diff))) / dof

    def __init__(self, x, y, encoders=None, technique=None):
        self.encoders = np.array(encoders)
        self.x = pd.DataFrame(x)
        self.y = y

        if encoders is None:
            self.encoders = np.array([SubsampleEncoder, DummyEncoder, ce.LeaveOneOutEncoder, ce.BinaryEncoder])

        y_shape = 1 if len(y.shape) == 1 else y.shape[1]

        min_size = 2 * (self.x.shape[1] + y_shape) + 1
        max_size = None

        if technique == 'GP' or technique == 'HDAGP':
            max_size = 4000

        self.metrics_checker = self.metrics(self.x.shape[1], min_size, max_size)

    def build_configuration(self, categorical_variables):
        if len(categorical_variables) == 0:
            return []

        categorical_variables = self.x.iloc[:, categorical_variables].columns.to_numpy(dtype=str)
        encoders_num = len(self.encoders)
        # steps = 3 * encoders_num
        coeff = [83.24192459, 0.00852112, 84.34873041, 492.72273704, 343.75608215,
                 1000000175.1594066]  # [83.24192459, 0.00852112, 84.34873041, 492.72273704, 343.75608215, 1000000175.1594066]
        probability_for_best = 0.7
        prediction = np.zeros_like(categorical_variables, dtype=int)
        probability_for_other = (1 - probability_for_best) / (len(prediction) - 1)
        configuration = self.__configure_encoders(prediction, categorical_variables)
        min_penalty = self.__calculate_penalty(configuration, coeff)

        while True:
            penalties = np.zeros_like(prediction, dtype=float)
            for i in range(len(prediction)):
                tmp = prediction[i]
                prediction[i] = (prediction[i] + 1) % encoders_num
                configuration = self.__configure_encoders(prediction, categorical_variables)
                penalties[i] = self.__calculate_penalty(configuration, coeff)
                prediction[i] = tmp

            i = np.argmin(penalties)
            if penalties[i] > min_penalty:
                break

            probabilities = np.full_like(prediction, probability_for_other, dtype=float)
            probabilities[i] = probability_for_best
            i = np.random.choice(len(prediction), p=probabilities)
            prediction[i] = (prediction[i] + 1) % encoders_num
            min_penalty = penalties[i]

        return self.__configure_encoders(prediction, categorical_variables)

    def __calculate_penalty(self, configuration, coeff):
        penalty = 0
        configuration = [encoder.fit(self.x, self.y) for encoder in configuration]

        initial_columns = set(self.x.columns)

        columns = set(self.x.columns)
        columns_to_delete = None

        for encoder in configuration:
            transformed = encoder.transform(self.x, self.y)
            sample_sizes = pd.Series([len(transformed)])
            transformed_columns = set(encoder.get_feature_names()) - initial_columns

            penalty += coeff[0] * self.metrics_checker.data_losing(
                transformed.loc[:, transformed_columns or encoder.cols].value_counts(),
                self.x.loc[:, encoder.cols].value_counts())

            if not transformed_columns:
                for col in encoder.cols:
                    penalty += coeff[1] * self.metrics_checker.diff_metric(transformed[col])

            if isinstance(encoder, SubsampleEncoder):
                columns_to_delete = set(encoder.cols)
                sample_sizes = self.x.groupby(encoder.cols).size()

            columns ^= set(encoder.get_feature_names())

            penalty += coeff[2] * self.metrics_checker.disribution_metric(sample_sizes)
            penalty += coeff[3] * self.metrics_checker.sample_count_metric(sample_sizes)
            penalty += coeff[4] * self.metrics_checker.elements_limit(sample_sizes)

        if columns_to_delete:
            columns -= columns_to_delete

        penalty += coeff[5] * self.metrics_checker.dimension_increase(len(columns))

        return penalty

    def __configure_encoders(self, encoder_numbers, cols=None):
        raw_classes = self.encoders[encoder_numbers]
        encoders_cols = {cls: cols[np.where(raw_classes == cls)[0]] for cls in raw_classes}
        result = sorted((encoder(cols=encoders_cols[encoder]) for encoder in encoders_cols),
                        key=lambda encoder: 1 if isinstance(encoder, SubsampleEncoder) else 0)

        if isinstance(result[-1], SubsampleEncoder):
            result.append(ce.OrdinalEncoder(cols=result[-1].cols))
            result[-1], result[-2] = result[-2], result[-1]

        return result

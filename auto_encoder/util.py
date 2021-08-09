import category_encoders as ce
from auto_encoder import DummyEncoder, SubsampleEncoder
import numpy as np
import pandas as pd
from collections import defaultdict



class ConfigurationManager:
    """
    Class, which build configuration for auto_encoder.Builder.
    >>> manager = ConfigurationManager(encoders=[SubsampleEncoder, DummyEncoder, ce.BinaryEncoder, ce.OneHotEncoder, ce.LeaveOneOutEncoder])
    >>> config = manager.build_configuration(categorical_encoders, dataset)
    """

    class metrics:

        def __init__(self, min_shape, min_elements):
            self.min_shape = min_shape
            self.min_elements = min_elements

        def std_limit(self, df):
            return np.log2((df.max() - df.min()) + 1) * 225

        def elements_limit(self, df):
            return np.count_nonzero(df.to_numpy() > self.min_elements)

        def shape_limit(self, df, column):
            return df.iloc[: column].unique()

    def __init__(self, encoders=None):
        self.encoders = encoders

        if self.encoders is None:
            self.encoders = [SubsampleEncoder, DummyEncoder, ce.HashingEncoder, ce.OneHotEncoder, ce.LeaveOneOutEncoder]

    def build_configuration(self, dataset, categorical_variables):
        dataset = pd.DataFrame(dataset)
        categorical_variables = dataset.iloc[:, categorical_variables].columns.to_numpy()

        def sorting_func(encoder):
            if isinstance(encoder, SubsampleEncoder):
                return 2
            elif isinstance(encoder, ce.OrdinalEncoder):
                return 1
            return 0

        metric_checker = self.metrics(min_shape=30, min_elements=2 * dataset.shape[1] + 4)

        configuration = defaultdict(list)

        for column in categorical_variables:
            scores = dict()

            for encoder in self.encoders:
                scores[encoder] = metric_checker.validate(encoder, dataset, column)

            optimal_encoder = min(scores, key=scores.get)
            if isinstance(optimal_encoder, SubsampleEncoder):
                configuration[ce.OrdinalEncoder].append(column)

            configuration[optimal_encoder].append(column)

        return sorted([encoder(cols=configuration[encoder]) for encoder in configuration], key=sorting_func)

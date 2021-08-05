from EncoderSubsample import EncoderSubsample
import pandas as pd
import numpy as np
from DummyEncoder import DummyEncoder
import category_encoders as ce


def to_dataframe(x):
    return pd.DataFrame(x, columns=[f'col{i}' for i in range(len(x[0]))])


def build_configuration(dataset, categorical_variables):
    def sorting_func(x):
        if isinstance(x[1], tuple) and EncoderSubsample in x[1]:
            return True
        elif not isinstance(x[1], EncoderSubsample):
            return False

        return True

    configuration = []
    std_limit = lambda s: np.log2((s.max() - s.min()) + 1) * 225
    dataset_size = len(dataset)

    if dataset_size < 15:
        return categorical_variables, DummyEncoder

    elements_in_group = max(15, 2 * dataset.shape[1] + 1)
    elements_count = {var: dataset[var].value_counts() for var in categorical_variables}
    elements_std = {var: elements_count[var].std() for var in categorical_variables}

    column = min(elements_std, key=elements_std.get)
    std = elements_std[column]
    groups_len = elements_count[column]
    columns_for_subsampling = []

    while std < std_limit(groups_len) and \
            all(groups_len.apply(lambda x: x >= elements_in_group)) and len(elements_std) > 0:
        columns_for_subsampling.append(column)
        elements_std.pop(column)

        column = min(elements_std, key=elements_std.get)
        group = dataset.groupby(columns_for_subsampling + [column])
        groups_len = group.size()
        std = groups_len.std()

    configuration.append((columns_for_subsampling, (ce.OrdinalEncoder, EncoderSubsample)))
    configuration.append((tuple(elements_std.keys()), DummyEncoder))

    return sorted(configuration, key=sorting_func)
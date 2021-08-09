import pandas as pd
from da.p7core_6_23_400 import gtapprox
from .encoders import SubsampleEncoder, DummyEncoder
from .models import Model, Submodels
import numpy as np
import category_encoders as ce


class Builder(gtapprox.Builder):

    @staticmethod
    def build_configuration(dataset, categorical_variables):

        def sorting_func(encoder_cadidate):
            if np.any(isinstance(_, SubsampleEncoder) for _ in encoder_cadidate[1]):
                return True

            return False

        configuration = []

        def std_limit(s):
            return np.log2((s.max() - s.min()) + 1) * 225

        dataset_size = len(dataset)

        if dataset_size < 15:
            return categorical_variables, DummyEncoder

        elements_in_group = max(15, 2 * dataset.shape[1] + 1)
        elements_count = {var: dataset.iloc[:, var].value_counts() for var in categorical_variables}
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

        configuration.append((columns_for_subsampling, (ce.OrdinalEncoder, SubsampleEncoder)))
        configuration.append((tuple(elements_std.keys()), DummyEncoder))

        return sorted(configuration, key=lambda encoder_candidate: np.any(isinstance(_, SubsampleEncoder) for _ in encoder_cadidate[1]))

    def build(self, x, y, encoding=None, **kwargs):
        if encoding is None:
            categorical_variables = kwargs.setdefault('options', {}).get("GTApprox/CategoricalVariables", [])
            encoding = self.build_configuration(pd.DataFrame(x), categorical_variables)

        if len(encoding) == 0:
            return super(Builder, self).build(x, y, **kwargs)

        encoder = encoding.pop(0).fit(x)
        if isinstance(encoder, SubsampleEncoder):
            submodels = Submodels(columns=encoder.cols)
            for columns, xs, ys in encoder.iterate_subsamples(x, y):
                print(columns, len(xs))
                submodels[columns] = self.build(xs, ys, encoding, **kwargs)
            return submodels
        else:
            return Model(encoder=encoder, model=self.build(encoder.transform(x, y), y, encoding=encoding, **kwargs))

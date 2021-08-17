from auto_encoder import DummyEncoder, SubsampleEncoder, Builder
from da.p7core_6_23_400 import gtapprox
from pydataset import data
import pandas as pd
import numpy as np
import category_encoders as ce
from itertools import permutations


def configure_encoder(encoders, encoder_number, cols=None):
    raw_classes = encoders[encoder_number]
    encoders_cols = {cls: cols[np.where(raw_classes == cls)[0]] for cls in raw_classes}
    return np.array([encoder(cols=encoders_cols[encoder]) for encoder in encoders_cols])


def get_class_name(cls_instances):
    return [f'{cls_instance.__class__.__name__}({", ".join(cls_instance.cols)})' for cls_instance in cls_instances]


def main():
    np.vectorize(get_class_name)

    dataset = data('diamonds')
    dataset = dataset.sample(frac=1, random_state=10000)
    categorical_variables = np.array([1, 2, 3])
    variable_names = dataset.columns[categorical_variables].to_numpy()
    dataset = ce.OrdinalEncoder(cols=variable_names).fit_transform(dataset)

    encoders = np.array([DummyEncoder, SubsampleEncoder, ce.LeaveOneOutEncoder, ce.BinaryEncoder])
    encoder_indexes = np.arange(start=0, stop=encoders.shape[0])
    indexes_grid = np.repeat(np.atleast_2d(encoder_indexes), categorical_variables.shape[0], axis=0)

    index_combinations = np.array(np.meshgrid(*indexes_grid)).T.reshape((-1, variable_names.shape[0]))
    encoder_combinations = np.array([configure_encoder(encoders, combination, variable_names) for combination in index_combinations], dtype=object)
    encoder_permutations = np.array([np.array([*permutations(combination)]) for combination in encoder_combinations], dtype=object)


    builder = Builder()

    dataset_for_testing = dataset.tail(len(dataset) // 100)
    dataset = dataset.iloc[:-len(dataset) // 100]
    y_columns = ['price']
    x_test = dataset_for_testing.drop(columns=y_columns)
    y_test = dataset_for_testing.loc[:, y_columns]

    education = dataset.sample(n=int(len(dataset) * 0.6), random_state=len(dataset))
    x, y = education.drop(columns=y_columns), education.loc[:, y_columns]

    options = {
        "GTApprox/CategoricalVariables": categorical_variables,
        "GTApprox/Technique": 'RSM',
        "/GTApprox/Binarization": True
    }

    errors = []
    print("Errors for all configurations")
    i = encoder_permutations.shape[0]
    for permutation in encoder_permutations:
        print(i)
        i -= 1
        rrms = [builder.build(x, y, encoding=list(encoding), options=options).validate(x_test, y_test)['RRMS'] for encoding in permutation]
        line = list(zip([get_class_name(encoding) for encoding in permutation], rrms))
        errors.append(line)

    errors = np.array(errors, dtype=object)
    print(errors)

    builder = gtapprox.Builder()

    print(builder.build(x, y, options=options).validate(x_test, y_test)['RRMS'])

if __name__ == '__main__':
    main()

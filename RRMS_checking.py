from auto_encoder import Builder
from da.p7core_6_23_400 import gtapprox
import category_encoders as ce
from pydataset import data
import matplotlib.pyplot as plt
import numpy as np
from sys import maxsize
from copy import deepcopy


def get_default_model_RRMS(x, y, x_test, y_test, categorical_variables, technique='RSM', binarization=True):
    variable_names = x.columns[categorical_variables].to_numpy()
    encoder = ce.OrdinalEncoder(variable_names).fit(x)
    x = encoder.transform(x)

    builder = gtapprox.Builder()
    options = {
        "GTApprox/CategoricalVariables": categorical_variables,
        "GTApprox/Technique": technique,
        "/GTApprox/Binarization": binarization
    }
    model = builder.build(x, y, options=options)

    x_test = encoder.transform(x_test)
    errors = model.validate(x_test, y_test)

    return errors['RRMS'][0]


def get_encoded_model_RRMS(x, y, x_test, y_test, categorical_variables, technique='RSM', binarization=True):
    builder = Builder()

    options = {
        "GTApprox/CategoricalVariables": categorical_variables,
        "GTApprox/Technique": technique,
        "/GTApprox/Binarization": binarization
    }

    model = builder.build(x, y, options=options)
    errors = model.validate(x_test, y_test)

    return errors['RRMS']


def main():
    tests = 100
    seed = tests // 5
    dataset = data('diamonds').sample(frac=1, random_state=seed)
    size = len(dataset)
    categorical_variables = [1, 2, 3]
    y_columns = ["price"]

    np.random.seed(seed)

    print(dataset)
    # test - 10%
    # education - 60-70%

    dataset_for_testing = dataset.tail(int(size * 0.1))
    dataset = dataset.iloc[:-int(size * 0.1)]
    x_test = dataset_for_testing.drop(columns=y_columns)
    y_test = dataset_for_testing.loc[:, y_columns]

    default_model_RRMS = np.zeros(shape=(tests,))
    encoded_model_RRMS = np.zeros(shape=(tests,))
    dataset_sizes = np.random.randint(size * 0.6, size * 0.7, size=tests)

    for i in range(tests):
        print(i, dataset_sizes[i])

        education = dataset.sample(n=dataset_sizes[i], random_state=i)
        x, y = education.drop(columns=y_columns), education.loc[:, y_columns]

        default_model_RRMS[i] = get_default_model_RRMS(
            deepcopy(x), deepcopy(y), deepcopy(x_test), deepcopy(y_test),
            categorical_variables
        )

        encoded_model_RRMS[i] = get_encoded_model_RRMS(
            deepcopy(x), deepcopy(y), deepcopy(x_test), deepcopy(y_test),
            categorical_variables
        )
        print(encoded_model_RRMS[i])

    fig, ax = plt.subplots(figsize=(10, 15))

    errors = np.unique(np.concatenate([default_model_RRMS, encoded_model_RRMS]))
    # errors = errors[~np.isnan(errors)]
    print(list(errors))
    encoded_model_count = np.array([np.count_nonzero(encoded_model_RRMS < error) for error in errors])
    default_model_count = np.array([np.count_nonzero(default_model_RRMS < error) for error in errors])

    ax.plot(errors, default_model_count, label="Стандартная модель")
    ax.plot(errors, encoded_model_count, label="Модель с кодированием")
    ax.grid()
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()

import numpy as np
from .builder import Builder
from da.p7core_6_23_400 import gtapprox
import category_encoders as ce
import matplotlib.pyplot as plt
import re
from datetime import timedelta


def convert_to_timedelta(timedelta_str):
    pattern = re.compile(r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)(\.(?P<microseconds>\d+))?')
    m = pattern.match(timedelta_str)

    params = {key: float(val) for key, val in m.groupdict(0).items()}

    return timedelta(**params)


def get_default_model_RRMS(x, y, x_test, y_test, options):
    variable_names = x.columns[options['GTApprox/CategoricalVariables']].to_numpy()
    encoder = ce.OrdinalEncoder(variable_names).fit(x)
    x = encoder.transform(x)

    builder = gtapprox.Builder()
    model = builder.build(x, y, options=options)
    building_time = convert_to_timedelta(model.details['Training Time']['Total']).total_seconds()

    x_test = encoder.transform(x_test)
    errors = model.validate(x_test, y_test)

    return errors['RRMS'][0], building_time


def get_encoded_model_RRMS(x, y, x_test, y_test, options):
    builder = Builder()

    model = builder.build(x, y, options=options)
    building_time = convert_to_timedelta(model.details['Training Time']['Total']).total_seconds()
    errors = model.validate(x_test, y_test)

    return errors['RRMS'], building_time


def build_plots(default_model_RRMS, encoded_model_RRMS, default_model_time, encoded_model_time):
    fig, axes = plt.subplots(2, 1, figsize=(10, 20))

    errors = np.unique(np.concatenate([default_model_RRMS, encoded_model_RRMS]))
    times = np.unique(np.concatenate([default_model_time, encoded_model_time]))

    encoded_model_count = np.array([np.count_nonzero(encoded_model_RRMS < error) for error in errors])
    default_model_count = np.array([np.count_nonzero(default_model_RRMS < error) for error in errors])

    encoded_model_time_count = np.array([np.count_nonzero(encoded_model_time < time) for time in times])
    default_model_time_count = np.array([np.count_nonzero(default_model_time < time) for time in times])

    fontsize = 20

    axes[0].plot(errors, default_model_count, label="gtapprox.Model")
    axes[0].plot(errors, encoded_model_count, label="Model with encoding")
    axes[0].set_xlabel('RRMS error', fontsize=fontsize)
    axes[0].set_ylabel('Model number', fontsize=fontsize)

    axes[1].plot(times, default_model_time_count, label="gtapprox.Model")
    axes[1].plot(times, encoded_model_time_count, label="Model with encoding")
    axes[1].set_xlabel('Training time', fontsize=fontsize)
    axes[1].set_ylabel('Model number', fontsize=fontsize)

    for ax in axes:
        ax.grid()
        ax.legend(fontsize=fontsize)

    plt.show()


def make_analysis(dataset, categorical_variables, y_columns, technique='RSM', binarization=True):
    """
    Script for generating performance analysis to see the difference between default model and model with encoding
    :param dataset: pd.Dataframe, which consist y_columns
    :param categorical_variables: numbers of columns with categorical variables
    :param y_columns: numbers of colums, which will be model outputs
    :param technique: string, which defines training technique. Default is RSM
    :param binarization: Flag to turn binarization off for technique, which uses binarization to encode categorical variables
    :return: None
    """
    tests = 100
    seed = tests // 4
    size = len(dataset)
    y_columns = dataset.columns[y_columns].to_numpy()
    print(y_columns)
    np.random.seed(seed)
    # test - 10%
    # education - 60-70%

    dataset_for_testing = dataset.tail(int(size * 0.1))
    dataset = dataset.iloc[:-int(size * 0.1)]
    x_test = dataset_for_testing.drop(columns=y_columns)
    y_test = dataset_for_testing.loc[:, y_columns]

    default_model_RRMS = np.zeros(shape=tests)
    encoded_model_RRMS = np.zeros(shape=tests)

    default_model_time = np.zeros(shape=tests, dtype=float)
    encoded_model_time = np.zeros(shape=tests, dtype=float)

    dataset_sizes = np.random.randint(size * 0.6, size * 0.7, size=tests)

    options = {
        "GTApprox/CategoricalVariables": categorical_variables,
        "GTApprox/Technique": technique,
        "/GTApprox/Binarization": binarization
    }

    for i in range(tests):
        print(i)
        education = dataset.sample(n=dataset_sizes[i], random_state=i)
        x, y = education.drop(columns=y_columns), education.loc[:, y_columns]

        default_model_RRMS[i], default_model_time[i] = get_default_model_RRMS(
            x, y, x_test, y_test, options.copy()
        )

        encoded_model_RRMS[i], encoded_model_time[i] = get_encoded_model_RRMS(
            x, y, x_test, y_test, options.copy()
        )

    build_plots(default_model_RRMS, encoded_model_RRMS, default_model_time, encoded_model_time)

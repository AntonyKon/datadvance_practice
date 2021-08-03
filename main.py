from Encoder import Encoder
from da.p7core_6_23_400 import gtapprox, loggers
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pydataset import data
import sys
from BuilderWithEncoding import BuilderWithEncoding
from EncoderSubsample import EncoderSubsample
import category_encoders as ce
from RRMS_checking import RRMS_calc
import matplotlib.pyplot as plt


# GBRT, RSM: binarization, submodels for gtapprox.builder and BuilderWithEncoding
# HDA: submodels for gtapprox.builder and BuilderWithEncoding

def filter_dataset(dataset, columns, group_size):
    # dataset.shape[0] / dataset[columns].drop_duplicates().shape[0]
    # unique_values = data[column_1].unique() # a, b, c
    # mask = np.zeros((x.shape[0], unique_values.size), dtype=bool)
    # x[mask[:, 0], column_1]
    return dataset.groupby(columns).filter(lambda x: len(x) > group_size)


def preprocess_dataset(dataset, columns, size):
    dataset = filter_dataset(dataset, ['cut', 'color', 'clarity'], 15)
    group_size = size // len(dataset.groupby(columns))
    assert group_size >= 15

    preprocessed_df = pd.DataFrame(columns=dataset.columns.values)
    for unique, df in dataset.groupby(columns):
        preprocessed_df = preprocessed_df.append(df.head(n=group_size), ignore_index=True)

    return preprocessed_df.sample(frac=1).reset_index(drop=True)


def main(technique=None):
    dataset = data("diamonds")
    print(dataset)

    if technique is None:
        technique = 'HDA'

    dataset["cut"] = LabelEncoder().fit_transform(dataset['cut'])
    dataset["color"] = LabelEncoder().fit_transform(dataset['color'])
    dataset["clarity"] = LabelEncoder().fit_transform(dataset['clarity'])
    stop = 25000

    dataset_for_education = preprocess_dataset(dataset, ['cut', 'color', 'clarity'], stop)  # dataset.loc[:stop, :]
    print(dataset_for_education)
    X = dataset_for_education.drop(columns=["price"])
    Y = dataset_for_education.price
    builder = gtapprox.Builder()
    # builder = BuilderWithEncoding()

    # builder.set_logger(loggers.StreamLogger())

    options = {
        "GTApprox/CategoricalVariables": [1, 2, 3],
        "GTApprox/Accelerator": 1,
        "GTApprox/Technique": technique,
        "/GTApprox/EncodingOptions": [(('cut', 'color', 'clarity'), (ce.OrdinalEncoder, EncoderSubsample)),

                                      ]  # 2*x.shape[1] + 1
        #     "GTApprox/GBRTShrinkage": 0.01,
        #     "GTApprox/GBRTSubsampleRatio": 0.1,
        #     "GTApprox/GBRTColsampleRatio": 0.6
    }

    model = builder.build(X, Y, options=options)
    # Y = np.random.random(100)
    # X = Y < 0.5
    #
    # builder.set_logger(loggers.StreamLogger(loglevel=loggers.LogLevel.INFO))
    #
    # model = builder.build(X, Y, options={'GTApprox/CategoricalVariables': [0], 'GTApprox/Technique': 'RSM'})

    dataset_for_testing = filter_dataset(dataset, ['cut', 'color', 'clarity'], 15).head(100)

    X_test = dataset_for_testing.drop(columns=['price'])
    Y_test = dataset_for_testing.price

    result = None

    dots = np.linspace(0, 99, 100)

    Y_predict = model.calc(X_test)

    print(Y_predict)

    fig, ax = plt.subplots(1, figsize=(10, 10))

    ax.scatter(dots, Y_test, label="Exact value")
    ax.plot(dots, Y_predict, 'ro--', label="Predicted value")

    ax.legend()
    ax.grid()

    plt.show()

    print(model.validate(X_test, Y_test))
    return result


if __name__ == '__main__':
    # with open('results.txt', 'w') as file:
    #     for technique in ['GBRT', 'RSM', 'HDA', 'GP', 'SGP']:
    #         file.write(f'{technique}: {main(technique)}\n')
    #         file.flush()
    main()

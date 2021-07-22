from Encoder import Encoder
from da.p7core_6_23_400 import gtapprox, loggers
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pydataset import data
import sys
from BuilderWithEncoding import BuilderWithEncoding
import matplotlib.pyplot as plt

def sort_sample(Y):
    if Y < 0.1:
        return 0
    elif Y < 0.5:
        return 1
    elif Y < 0.9:
        return 2
    else:
        return 3

def main():
    diamonds = data("diamonds")
    diamonds["cut"] = LabelEncoder().fit_transform(diamonds['cut'])
    diamonds["color"] = LabelEncoder().fit_transform(diamonds['color'])
    diamonds["clarity"] = LabelEncoder().fit_transform(diamonds['clarity'])
    stop = len(diamonds) // 4
    diamonds_for_education = diamonds.loc[:stop, :]
    X = diamonds_for_education.drop(columns=["price"])
    Y = diamonds_for_education.price.values
    builder = BuilderWithEncoding()

    builder.set_logger(loggers.StreamLogger())

    options = {
        "GTApprox/CategoricalVariables": [1, 2, 3],
        "GTApprox/Accelerator": 1,
        "GTApprox/Technique": "RSM"
        #     "GTApprox/GBRTShrinkage": 0.01,
        #     "GTApprox/GBRTSubsampleRatio": 0.1,
        #     "GTApprox/GBRTColsampleRatio": 0.6
    }

    # X.to_numpy()
    model = builder.build(X, Y, options=options)
    #
    # np.vectorize(sort_sample)
    #
    # Y = np.random.random(100)
    # X = Y < 0.5
    #
    # builder = gtapprox.Builder()
    #
    # builder.set_logger(loggers.StreamLogger())
    #
    # model = builder.build(X, Y, options={'GTApprox/CategoricalVariables': [0], 'GTApprox/Technique': 'RSM'})

    diamonds_for_checking = Encoder('binary', cols=diamonds.columns.values[1:4]).fit_transform(diamonds.loc[stop:, :])
    n = 100

    tests = 1000
    RRMS_mean = 0
    for i in range(tests):
        rows = [row for index, row in diamonds_for_checking.sample(n=n).iterrows()]
        #     predicted_values = [model.calc(row.drop(labels=['price']).to_numpy())[0] for row in rows]
        exact_values = [row.at['price'] for row in rows]
        RRMS_mean += model.validate([row.drop(labels=['price']).to_numpy() for row in rows], exact_values)['RRMS'][0]
        sys.stdout.write(str(i) + '\n')
    #     print(i)
    print(RRMS_mean / tests)




if __name__ == '__main__':
    main()
from Encoder import Encoder
from da.p7core_6_23_400 import gtapprox, loggers
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pydataset import data
import sys
from BuilderWithEncoding import BuilderWithEncoding
from RRMS_checking import RRMS_calc
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
    builder = gtapprox.Builder()

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

    diamonds_for_checking = diamonds.loc[stop:, :] # Encoder('binary', cols=diamonds.columns.values[1:4]).
    n = 1000

    tests = 100

    print(RRMS_calc(model, diamonds_for_checking, n, tests))




if __name__ == '__main__':
    main()
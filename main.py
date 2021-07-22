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
        "GTApprox/Technique": "GBRT"
        #     "GTApprox/GBRTShrinkage": 0.01,
        #     "GTApprox/GBRTSubsampleRatio": 0.1,
        #     "GTApprox/GBRTColsampleRatio": 0.6
    }

    model = builder.build_smart(X, Y, options=options)
    # Y = np.random.random(100)
    # X = Y < 0.5
    #
    # builder = gtapprox.Builder()
    # builder.set_logger(loggers.StreamLogger())
    #
    # model = builder.build(X, Y, options={'GTApprox/CategoricalVariables': [0], 'GTApprox/Technique': 'RSM'})

    if isinstance(builder, BuilderWithEncoding):
        diamonds_for_checking = Encoder('binary', cols=diamonds.columns.values[1:4]).fit_transform(diamonds.loc[stop:, :])
    else:
        diamonds_for_checking = diamonds.loc[stop:, :]

    n = 1000
    tests = 1000
    print(RRMS_calc(model, diamonds_for_checking, n, tests))


if __name__ == '__main__':
    main()

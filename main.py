import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from da.p7core_6_23_400 import gtapprox, loggers
from pydataset import data

import category_encoders as ce
import auto_encoder

dataset = data("diamonds")
x = dataset.iloc[::100, :6]
y = dataset.iloc[::100, [6]]

print(type(x), type(y))


model = auto_encoder.Builder().build(x, y, encoding=[
                                             auto_encoder.DummyEncoder(cols=['cut', 'color', 'clarity'])
                                           ],
                                           options={
                                             "GTApprox/Accelerator": 1,
                                             "GTApprox/Technique": 'GBRT',
                                           })

x_encoded = ce.OrdinalEncoder(cols=['cut', 'color', 'clarity']).fit_transform(x)
model2 = gtapprox.Builder().build(x_encoded, y, options={
                                                   "GTApprox/CategoricalVariables": [1, 2, 3],
                                                   "GTApprox/Accelerator": 1,
                                                   "GTApprox/Technique": 'GBRT',
                                                 })
print(model.calc(x))

assert np.abs(model.calc(x) - model2.calc(x_encoded)).max() < 1e-16
assert np.abs(model.validate(x, y)['RRMS'] - model2.validate(x_encoded, y)['RRMS'][0]) < 1e-16

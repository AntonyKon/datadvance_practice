import numpy as np
import pandas as pd
from collections import namedtuple

class BaseModel(object):
    def calc(self, x):
        raise NotImplementedError()

    def validate(self, x, y_true):
        raise NotImplementedError()


class Submodels(BaseModel):
    def __init__(self, models=None, columns=None):
        '''

        :param models: dict, where key = values of special columns, value = model
        :param columns: tuple of column names
        '''

        self.models = models
        self.columns = columns

    def validate(self, x, y_true):
        y_predict = []
        for x_input in x.itertuples():
            values = tuple(x_input[self.columns[0]: self.columns[-1]])
            model = self.models.get(values, None)
            if model is not None:
                y_predict.append(self.models[values].calc(x))
            else:
                y_predict.append(None)

        y_predict = np.array(y_predict)
        return np.sqrt(np.sum((y_true - y_predict) ** 2)) / np.std(y_true, ddof=1)

    def calc(self, x):
        if isinstance(x, pd.DataFrame):
            y = []
            for index, x_input in x.iterrows():
                values = tuple(x_input[self.columns[0]: self.columns[-1]])
                model = self.models.get(values, None)
                if model is not None:
                    res = model.calc(x_input)
                    if len(res) == 1:
                        res = res[0]

                    y.append(res)
                else:
                    y.append(None)
            return y
        elif isinstance(x, pd.Series):
            values = tuple(x.loc[self.columns[0]: self.columns[-1]])
            model = self.models.get(values, None)
            if model is not None:
                return model.calc(x)
        return None

    def __str__(self):
        return '\n'.join([f'{columns}: {model}' for columns, model in self.models.items()])


class Model(BaseModel):
    def __init__(self, model=None, encoder=None):
        self.model = model
        self.encoder = encoder

    def calc(self, x):
        if isinstance(x, pd.Series):
            x = x.to_frame().T
        x = self.encoder.transform(x)
        res = self.model.calc(x)

        if len(res) == 1:
            res = res[0]
        return res

    def validate(self, x, y_true):
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true = y_true.to_numpy()
        x = self.encoder.transform(x)
        y_predict = np.array([self.model.calc(x_input) for index, x_input in x.iterrows()])

        return np.sqrt(np.mean((y_predict - y_true) ** 2)) / np.std(y_true, ddof=1)

    def __str__(self):
        return f'{self.encoder}: {self.model}'


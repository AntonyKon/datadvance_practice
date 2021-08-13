import numpy as np
import pandas as pd


class BaseModel(object):

    def calc(self, x):
        raise NotImplementedError()

    def validate(self, x, y_true):
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.to_numpy()
        elif isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy().reshape((-1, 1))
        elif isinstance(y_true, (list, tuple)):
            y_true = np.array(y_true)
        elif isinstance(y_true, dict):
            y_true = np.array(list(y_true.values()))

        y_predict = np.array(self.calc(x))
        is_nan = np.isnan(y_predict)
        absolute_errors = np.abs(y_predict - y_true)[~is_nan]
        rms = np.sqrt(np.mean(absolute_errors ** 2))
        return {
            'RMS': rms,
            'RRMS': rms / np.std(y_true, ddof=1),
            'Mean': np.mean(absolute_errors),
            'Max': np.max(absolute_errors),
            'R^2': 1 - (rms ** 2) / np.var(y_true, ddof=1),
            'Median': np.median(absolute_errors),
        }


class Submodels(BaseModel):

    def __init__(self, models=None, columns=None):
        '''
    :param models: dict, where key = values of special columns, value = model
    :param columns: tuple of column names
    '''
        self.models = models or {}
        self.columns = columns or []

    def __getitem__(self, key):
        return self.models.get(key)

    def __setitem__(self, key, model):
        self.models[key] = model

    def calc(self, x):
        x = pd.DataFrame(x)
        y = pd.DataFrame(index=x.index, columns=['output'])
        for values, x_input in x.groupby(self.columns):
            if values in self.models:
                y.loc[x_input.index] = self.models[values].calc(x_input)
        return y.to_numpy(dtype=float)

    def __str__(self):
        return '\n'.join([f'{columns}: {model}' for columns, model in self.models.items()])


class Model(BaseModel):

    def __init__(self, model=None, encoder=None):
        self.model = model
        self.encoder = encoder

    def calc(self, x):
        x = self.encoder.transform(pd.DataFrame(x))
        return self.model.calc(x)

    def __str__(self):
        return f'{self.encoder}: {self.model}'

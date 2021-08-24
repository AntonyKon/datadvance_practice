import numpy as np
import pandas as pd
from datetime import datetime


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

    @property
    def details(self):
        raise NotImplementedError()


class Submodels(BaseModel):

    @property
    def details(self):
        """
        Implemetation only for details['Training Time']
        :return: dict with details of training models (only time yet)
        """

        details = dict()
        details['Training Time'] = dict()  # {'Start': '2021-08-24 14:29:53.750659', 'Finish': '2021-08-24 14:29:53.915443', 'Total': '0:00:00.164784'}

        for model in self.models.values():
            start_time = details['Training Time'].get('Start', model.details['Training Time']['Start'])
            details['Training Time']['Start'] = str(min(
                datetime.fromisoformat(start_time), datetime.fromisoformat(model.details['Training Time']['Start'])
            ))

            finish_time = details['Training Time'].get('Start', model.details['Training Time']['Finish'])
            details['Training Time']['Finish'] = str(max(
                datetime.fromisoformat(finish_time), datetime.fromisoformat(model.details['Training Time']['Finish'])
            ))
        details['Training Time']['Total'] = str(datetime.fromisoformat(details['Training Time']['Finish']) - datetime.fromisoformat(details['Training Time']['Start']))

        return details

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

    @property
    def details(self):
        return self.model.details

    def __init__(self, model=None, encoder=None):
        self.model = model
        self.encoder = encoder

    def calc(self, x):
        x = self.encoder.transform(pd.DataFrame(x))
        return self.model.calc(x)

    def __str__(self):
        return f'{self.encoder}: {self.model}'

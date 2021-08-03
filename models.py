import numpy as np
import pandas as pd


class BaseModel(object):
    def calc(self, x):
        raise NotImplementedError()

    def validate(self, x, y_true):
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true = y_true.to_numpy()
        elif isinstance(y_true, (list, tuple)):
            y_true = np.array(y_true)
        elif isinstance(y_true, dict):
            y_true = np.array(list(y_true.values()))

        y_predict = np.array(self.calc(x))
        absolute_errors = np.abs(y_predict - y_true)

        errors = dict()

        errors['RRMS'] = np.sqrt(np.mean(absolute_errors ** 2)) / np.std(y_true, ddof=1)
        errors['RMS'] = np.sqrt(np.mean(absolute_errors) ** 2)
        errors['Mean'] = np.mean(absolute_errors)
        errors['Max'] = np.max(absolute_errors)
        errors['R^2'] = 1 - (errors['RMS'] ** 2) / np.var(y_predict)
        errors['Median'] = np.median(absolute_errors)

        return errors


class Submodels(BaseModel):
    def __init__(self, models=None, columns=None):
        '''

        :param models: dict, where key = values of special columns, value = model
        :param columns: tuple of column names
        '''

        self.models = models
        self.columns = columns

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
                    y.append(np.nan)
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

    def __str__(self):
        return f'{self.encoder}: {self.model}'

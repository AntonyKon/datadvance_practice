import category_encoders as ce
import pandas as pd


class Encoder:
    def __init__(self, encoding_type, cols=None):
        self.encoding_type = encoding_type
        self.columns = cols

    def __get_columns(self, X):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.values
        elif isinstance(X, pd.Series):
            self.columns = [X.name]

    def fit(self, X):
        if self.columns is None:
            self.__get_columns(X)

        return self

    def transform(self, X):
        if self.encoding_type == 'binary':
            return self.__binary_transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __binary_transform(self, X):
        X_copy = X.copy()
        encoder = ce.OneHotEncoder(cols=self.columns)
        transformed = encoder.fit(X_copy).transform(X_copy)

        return transformed

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
            return self.__transform_with_class(X, ce.OneHotEncoder)
        elif self.encoding_type == 'mean':
            pass

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __transform_with_class(self, X, SpecialEncoder):
        X_copy = X.copy()
        encoder = SpecialEncoder(cols=self.columns)
        transformed = encoder.fit(X_copy).transform(X_copy)

        return transformed

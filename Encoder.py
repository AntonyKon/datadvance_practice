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

    def fit(self, X, Y=[]):
        if self.columns is None:
            self.__get_columns(X)

        return self

    def transform(self, X, Y):
        if self.encoding_type == 'binary':
            return self.__transform_with_class(X, ce.OneHotEncoder)
        elif self.encoding_type == 'leave_one_out':
            return self.__transform_with_class_supervised(X, Y, ce.LeaveOneOutEncoder)

    def fit_transform(self, X, Y=[]):
        return self.fit(X, Y).transform(X, Y)

    def __transform_with_class(self, X, SpecialEncoder):
        X_copy = X.copy()
        encoder = SpecialEncoder(cols=self.columns)
        transformed = encoder.fit(X_copy).transform(X_copy)

        return transformed

    def __transform_with_class_supervised(self, X, Y, SpecialEncoder):
        X_copy = X.copy()
        encoder = SpecialEncoder(cols=self.columns)
        transformed = encoder.fit(X_copy, Y).transform(X_copy)

        return transformed

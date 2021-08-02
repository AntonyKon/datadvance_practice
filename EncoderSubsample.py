import category_encoders as  ce
import pandas as pd


class EncoderSubsample(ce.OrdinalEncoder):
    def transform_to_datasets(self, X):
        for columns, subsample in X.groupby(self.cols):
            yield columns, subsample

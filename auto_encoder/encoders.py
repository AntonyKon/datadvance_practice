import category_encoders


class DummyEncoder(category_encoders.OneHotEncoder):

    def generate_mapping(self):
        mapping = super(DummyEncoder, self).generate_mapping()

        for switch in mapping:
            switch['mapping'] = switch['mapping'].iloc[:, :-1]

            if self.handle_unknown == 'value':
                switch['mapping'].loc[-1] = 1

            if self.handle_missing == 'value':
                switch['mapping'].loc[-2] = 1

        return mapping


class SubsampleEncoder(category_encoders.OrdinalEncoder):

    def iterate_subsamples(self, x, y):
        for columns, xs in x.groupby(self.cols):
            yield columns, xs, y.loc[xs.index]

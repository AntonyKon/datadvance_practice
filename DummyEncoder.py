import category_encoders as ce
import numpy as np
import pandas as pd


class DummyEncoder(ce.OneHotEncoder):
    def generate_mapping(self):
        mapping = []
        found_column_counts = {}

        for switch in self.ordinal_encoder.mapping:
            col = switch.get('col')
            values = switch.get('mapping').copy(deep=True)

            if self.handle_missing == 'value':
                values = values[values > 0]

            if len(values) == 0:
                continue

            index = []
            new_columns = []

            for cat_name, class_ in values.iteritems():
                if self.use_cat_names:
                    n_col_name = str(col) + '_%s' % (cat_name,)
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                else:
                    n_col_name = str(col) + '_%s' % (class_,)

                index.append(class_)
                new_columns.append(n_col_name)

            if self.handle_unknown == 'indicator':
                n_col_name = str(col) + '_%s' % (-1,)
                if self.use_cat_names:
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                new_columns.append(n_col_name)
                index.append(-1)

            base_matrix = np.eye(N=len(index), M=len(index)-1, k=-1, dtype=np.int) # changed shape of matrix
            base_df = pd.DataFrame(data=base_matrix, columns=new_columns[:-1], index=index) # all columns except las

            if self.handle_unknown == 'value':
                base_df.loc[-1] = 1
            elif self.handle_unknown == 'return_nan':
                base_df.loc[-1] = np.nan

            if self.handle_missing == 'return_nan':
                base_df.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                base_df.loc[-2] = 1

            mapping.append({'col': col, 'mapping': base_df})

        return mapping

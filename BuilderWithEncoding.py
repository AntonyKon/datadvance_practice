from da.p7core_6_23_400 import gtapprox
from EncoderSubsample import EncoderSubsample
from models import Model, Submodels
from copy import deepcopy
from operator import itemgetter
import pandas as pd
from preprocessing import build_configuration, to_dataframe


# [{('col1', 'col2', 'col3'): EncoderSubsample}]
#
#
# [{('col1', 'col2'): EncoderSubsample}, {('col3'): Encoder1}]
# [(('col1', 'col2', 'col3'), (ce.OrdinalEncoder, EncoderSubsample)), {('col4'): EncoderSubsample}


class BuilderWithEncoding(gtapprox.Builder):

    def __build_tree_models(self, x, y, **kwargs):
        encoding_options = kwargs['options'].get("/GTApprox/EncodingOptions", [])

        if encoding_options:
            columns, Encoder = encoding_options[0]

            if isinstance(Encoder, (tuple, list)):
                i = kwargs.get('encoder_index', 0)
                Encoder = Encoder[i]
                kwargs['encoder_index'] = i + 1

                if i + 1 == len(encoding_options[0][1]):
                    encoding_options.pop(0)
            else:
                encoding_options.pop(0)
            encoder = Encoder(cols=columns).fit(x)

            if isinstance(encoder, EncoderSubsample):
                submodels = Submodels()
                submodels.models = {}
                submodels.columns = columns

                for columns, group in encoder.transform_to_datasets(x):
                    print(columns, len(group))
                    indexes = group.index.to_numpy()
                    y_group = itemgetter(indexes)(y)

                    submodels.models[columns] = self.__build_tree_models(group, y_group, **deepcopy(kwargs))

                return submodels
            else:
                model = Model()

                x_transformed = encoder.transform(x, y)
                model.encoder = encoder
                model.model = self.__build_tree_models(x_transformed, y, **kwargs)

                return model
        else:
            kwargs.pop('encoder_index', None)
            return super().build(x, y, **kwargs)

    def build(self, x, y, options=None, outputNoiseVariance=None, comment=None, weights=None, initial_model=None,
              annotations=None, x_meta=None, y_meta=None):
        if not isinstance(x, pd.DataFrame):
            x = to_dataframe(x)

        if options:
            categorical_variables = options.get("GTApprox/CategoricalVariables", [])

            if categorical_variables:
                if not options.get("/GTApprox/EncodingOptions", False):
                    options["/GTApprox/EncodingOptions"] = build_configuration(x, itemgetter(categorical_variables)(x.columns.to_numpy()))
                print(options["/GTApprox/EncodingOptions"])

                del options["GTApprox/CategoricalVariables"]
                return self.__build_tree_models(x, y, options=options, outputNoiseVariance=outputNoiseVariance,
                                                comment=comment, weights=weights, initial_model=initial_model,
                                                annotations=annotations, x_meta=x_meta, y_meta=y_meta)

        return super().build(x, y, options, outputNoiseVariance, comment, weights, initial_model,
                             annotations, x_meta, y_meta)

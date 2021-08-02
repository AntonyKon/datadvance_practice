from da.p7core_6_23_400 import gtapprox
from EncoderSubsample import EncoderSubsample
from models import Model, Submodels
from copy import deepcopy


# [{('col1', 'col2', 'col3'): EncoderSubsample}]
#
#
# [{('col1', 'col2'): EncoderSubsample}, {('col3'): Encoder1}]
# [{('col3'): Encoder1}, {('col1', 'col2'): EncoderSubsample}, {('col4'): EncoderSubsample}]

class BuilderWithEncoding(gtapprox.Builder):
    def __build_tree_models(self, x, y, **kwargs):
        encoding_options = kwargs['options'].get("/GTApprox/EncodingOptions", [])

        if encoding_options:
            columns, Encoder = encoding_options[0].popitem()
            encoder = Encoder(cols=columns).fit(x)

            kwargs['options']["/GTApprox/EncodingOptions"].pop(0)

            if isinstance(encoder, EncoderSubsample):
                submodels = Submodels()
                submodels.models = {}
                submodels.columns = columns

                for columns, group in encoder.transform_to_datasets(x):
                    print(columns, len(group))
                    indexes = group.index.tolist()
                    y_group = [y[i] for i in indexes]

                    submodels.models[columns] = self.__build_tree_models(group, y_group, **deepcopy(kwargs))

                print(submodels)
                return submodels
            else:
                model = Model()

                x_transformed = encoder.fit_transform(x, y)
                model.encoder = encoder
                model.model = self.__build_tree_models(x_transformed, y, **kwargs)

                return model
        else:
            return super().build(x, y, **kwargs)

    def build(self, x, y, options=None, outputNoiseVariance=None, comment=None, weights=None, initial_model=None,
              annotations=None, x_meta=None, y_meta=None):
        if options:
            categorical_variables = options.get("GTApprox/CategoricalVariables", [])

            if categorical_variables:
                del options["GTApprox/CategoricalVariables"]
                return self.__build_tree_models(x, y, options=options, outputNoiseVariance=outputNoiseVariance,
                                                comment=comment, weights=weights, initial_model=initial_model,
                                                annotations=annotations, x_meta=x_meta, y_meta=y_meta)

        return super().build(x, y, options, outputNoiseVariance, comment, weights, initial_model,
                             annotations, x_meta, y_meta)

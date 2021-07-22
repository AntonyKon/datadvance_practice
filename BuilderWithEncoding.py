from da.p7core_6_23_400 import gtapprox
from Encoder import Encoder


class BuilderWithEncoding(gtapprox.Builder):
    def __encode_categorial_variables(self, x, y, options):
        if options:
            categorial_variables = options.get("GTApprox/CategoricalVariables", [])
            if categorial_variables:
                encoder = Encoder('leave_one_out',
                                  cols=x.columns.values[categorial_variables[0]: categorial_variables[-1] + 1])
                x = encoder.fit_transform(x, y)

            options["GTApprox/CategoricalVariables"] = []

        return x, options

    def build(self, x, y, options=None, outputNoiseVariance=None, comment=None, weights=None, initial_model=None,
              annotations=None, x_meta=None, y_meta=None):
        x, options = self.__encode_categorial_variables(x, y, options)

        return super().build(x, y, options, outputNoiseVariance, comment, weights, initial_model, annotations, x_meta,
                             y_meta)

    def build_smart(self, x, y, options=None, outputNoiseVariance=None, comment=None, weights=None, initial_model=None,
                    hints=None, x_test=None, y_test=None, annotations=None, x_meta=None, y_meta=None):
        x, options = self.__encode_categorial_variables(x, y, options)

        return super().build_smart(x, y, options, outputNoiseVariance, comment, weights, initial_model,
                                   hints, x_test, y_test, annotations, x_meta, y_meta)

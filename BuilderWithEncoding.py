from da.p7core_6_23_400 import gtapprox
from Encoder import Encoder


class BuilderWithEncoding(gtapprox.Builder):
    def build(self, x, y, options=None, outputNoiseVariance=None, comment=None, weights=None, initial_model=None, annotations=None, x_meta=None, y_meta=None):
        if options:
            categorial_variables = options.get("GTApprox/CategoricalVariables", [])
            if categorial_variables:
                encoder = Encoder('binary', cols=x.columns.values[categorial_variables[0]: categorial_variables[-1] + 1])
                x = encoder.fit_transform(x)

        print(x.shape)

        options["GTApprox/CategoricalVariables"] = []

        return super().build(x, y, options, outputNoiseVariance, comment, weights, initial_model, annotations, x_meta, y_meta)
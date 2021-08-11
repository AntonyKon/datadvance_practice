import pandas as pd
from da.p7core_6_23_400 import gtapprox
from .encoders import SubsampleEncoder
from .models import Model, Submodels
from .util import ConfigurationManager
import numpy as np
import category_encoders as ce


class Builder(gtapprox.Builder):

    def build(self, x, y, encoding=None, **kwargs):
        if encoding is None:
            categorical_variables = kwargs.setdefault('options', {}).get("GTApprox/CategoricalVariables", [])
            manager = ConfigurationManager(x, y)
            encoding = manager.build_configuration(categorical_variables)
            print(encoding)

        if len(encoding) == 0:
            return super(Builder, self).build(x, y, **kwargs)

        encoder = encoding.pop(0).fit(x, y)
        if isinstance(encoder, SubsampleEncoder):
            submodels = Submodels(columns=encoder.cols)
            for columns, xs, ys in encoder.iterate_subsamples(x, y):
                # print(columns, len(xs))
                submodels[columns] = self.build(xs, ys, encoding, **kwargs)
            return submodels
        else:
            return Model(encoder=encoder, model=self.build(encoder.transform(x, y), y, encoding=encoding, **kwargs))

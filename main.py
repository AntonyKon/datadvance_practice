from auto_encoder import convert_to_timedelta, Builder, plot
from pydataset import data
import numpy as np
import category_encoders as ce
from da.p7core_6_23_400 import gtapprox


def get_default_model_RRMS(x, y, x_test, y_test, options):
    variable_names = x.columns[options['GTApprox/CategoricalVariables']].to_numpy()
    encoder = ce.OrdinalEncoder(variable_names).fit(x)
    x = encoder.transform(x)

    builder = gtapprox.Builder()
    model = builder.build(x, y, options=options)
    building_time = convert_to_timedelta(model.details['Training Time']['Total']).total_seconds()

    x_test = encoder.transform(x_test)
    errors = model.validate(x_test, y_test)

    return errors['RRMS'][0], building_time


def get_encoded_model_RRMS(x, y, x_test, y_test, options):
    builder = Builder()

    model = builder.build(x, y, options=options)
    building_time = convert_to_timedelta(model.details['Training Time']['Total']).total_seconds()
    errors = model.validate(x_test, y_test)

    return errors['RRMS'], building_time



dataset = data('BudgetFood').sample(frac=1, random_state=1)
print(dataset)
categorical_variables = [4, 5] # indexes of categorical variables
variable_names = dataset.columns[categorical_variables].to_numpy()
print(variable_names)
y_columns = [1] # indexes of output columns
binarization = True
technique = 'RSM'
tests = 100

seed = tests // 4
size = len(dataset)
y_columns = dataset.columns[y_columns].to_numpy()
np.random.seed(seed)
# test - 10%
# train - 60-70%

dataset_for_testing = dataset.tail(int(size * 0.1))
dataset = dataset.iloc[:-int(size * 0.1)]
x_test = dataset_for_testing.drop(columns=y_columns)
y_test = dataset_for_testing.loc[:, y_columns]

default_model_RRMS = np.zeros(shape=tests)
encoded_model_RRMS = np.zeros(shape=tests)

default_model_time = np.zeros(shape=tests, dtype=float)
encoded_model_time = np.zeros(shape=tests, dtype=float)

dataset_sizes = np.random.randint(size * 0.6, size * 0.7, size=tests)

options = {
    "GTApprox/CategoricalVariables": categorical_variables,
    "GTApprox/Technique": technique,
    "/GTApprox/Binarization": binarization
}

builder = gtapprox.Builder()
builder2 = Builder()

for i in range(tests):
    print(i)
    train = dataset.sample(n=dataset_sizes[i], random_state=i)
    x, y = train.drop(columns=y_columns), train.loc[:, y_columns]
    options["GTApprox/CategoricalVariables"] = x.columns.get_indexer_for(variable_names)
    encoder = ce.OrdinalEncoder(variable_names).fit(x)

    model = builder.build(encoder.transform(x), y, options=options.copy())
    default_model_time[i] = convert_to_timedelta(model.details['Training Time']['Total']).total_seconds()
    default_model_RRMS[i] = model.validate(encoder.transform(x_test), y_test)['RRMS'][0]

    model2 = builder2.build(x, y, options=options.copy())
    encoded_model_time[i] = convert_to_timedelta(model2.details['Training Time']['Total']).total_seconds()
    default_model_RRMS[i] = model2.validate(x_test, y_test)['RRMS']

plot(default_model_RRMS, encoded_model_RRMS, default_model_time, encoded_model_time)

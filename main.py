from auto_encoder import make_analysis
from pydataset import data


dataset = data('BudgetFood').sample(frac=1, random_state=1)
print(dataset)
categorical_variables = [3, 4]
y_columns = [1]
binarization = True
technique = 'RSM'

make_analysis(dataset, categorical_variables, y_columns, technique, binarization)

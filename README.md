# Feature Scaling

bringing all features (input variables) into a similar range so that no single feature dominates others just because it has bigger numbers.

# some code of X y split code in features 

In machine learning, we always separate inputs and outputs:

X → features (inputs)

y → target (what we want to predict)

import pandas as pd

data = pd.DataFrame({
    'Age': [25, 30, 22, 35, 28],
    'Income': [50000, 60000, 45000, 80000, 52000],
    'LoanApproval': [1, 0, 1, 0, 1]
})

X = data.drop(columns=['LoanApproval'])

# y = target column
y = data['LoanApproval']

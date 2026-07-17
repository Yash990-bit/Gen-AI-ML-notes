# Feature Scaling
Feature Scaling is the process of bringing all features (input variables) into a similar range so that no single feature dominates others just because it has larger values. It helps many machine learning algorithms perform better and converge faster.
X and y Split
In machine learning, we always separate inputs and outputs:

X → Features (input variables)

y → Target (the value we want to predict)
import pandas as pd

data = pd.DataFrame({
    'Age': [25, 30, 22, 35, 28],
    'Income': [50000, 60000, 45000, 80000, 52000],
    'LoanApproval': [1, 0, 1, 0, 1]
})

# X = input features
X = data.drop(columns=['LoanApproval'])

# y = target column
y = data['LoanApproval']
# Bagging (Bootstrap Aggregating)

Bagging is an ensemble learning technique that trains multiple models on different random subsets of the training data. The final prediction is made by combining the predictions of all models (e.g., majority voting for classification or averaging for regression). It improves accuracy and reduces overfitting.
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

bagging.fit(X, y)
# Voting Classifier

A Voting Classifier combines the predictions of multiple machine learning models. The final prediction is based on majority voting (hard voting) or the average predicted probabilities (soft voting), often resulting in better performance than a single model.
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier())
    ],
    voting='hard'
)

voting.fit(X, y)
#XGBoost (Extreme Gradient Boosting)

XGBoost is a powerful gradient boosting algorithm that builds decision trees sequentially. Each new tree corrects the errors made by the previous trees, resulting in high accuracy, faster training, and strong performance on structured datasets.
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=42)

xgb.fit(X, y)
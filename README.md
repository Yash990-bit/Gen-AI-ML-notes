Feature Scaling

Feature Scaling is the process of bringing all features (input variables) into a similar range so that no single feature dominates others just because it has larger values. It helps many machine learning algorithms perform better and converge faster.

X and y Split

In machine learning, we always separate inputs and outputs:

X → Features (Input Variables)
y → Target (Output Variable)
import pandas as pd

data = pd.DataFrame({
    'Age': [25, 30, 22, 35, 28],
    'Income': [50000, 60000, 45000, 80000, 52000],
    'LoanApproval': [1, 0, 1, 0, 1]
})

# Input Features
X = data.drop(columns=['LoanApproval'])

# Target Variable
y = data['LoanApproval']
Bagging (Bootstrap Aggregating)

Bagging is an ensemble learning technique that trains multiple models on different random subsets of the training data. The final prediction is made by combining the predictions of all models.

Classification: Majority Voting
Regression: Average Prediction

It reduces variance and helps prevent overfitting.

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

bagging.fit(X, y)
Voting Classifier

A Voting Classifier combines multiple machine learning models.

Hard Voting: Majority Vote
Soft Voting: Average of Prediction Probabilities
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
XGBoost (Extreme Gradient Boosting)

XGBoost is an optimized implementation of Gradient Boosting. It builds trees sequentially, where every new tree tries to correct the mistakes made by the previous trees.

Unlike Random Forest (where trees are independent), XGBoost learns from the errors of earlier trees.

Mathematical Working of XGBoost

Suppose we have a dataset

D={(x
i
	​

,y
i
	​

)}
i=1
n
	​


where

x
i
	​

 = input features
y
i
	​

 = actual target
Step 1. Initial Prediction

The model starts with a constant prediction.

For regression,

y
^
	​

(0)
=mean(y)

For classification,

y
^
	​

(0)
=log(
1−p
p
	​

)

where

p=
total samples
number of positive samples
	​

Step 2. Prediction After m Trees

After adding m trees,

y
^
	​

i
	​

=
k=1
∑
m
	​

f
k
	​

(x
i
	​

)
	​


where

f
k
	​

 = kth decision tree
Each tree predicts only a small correction.
Step 3. Objective Function

XGBoost minimizes

Obj=
i=1
∑
n
	​

L(y
i
	​

,
y
^
	​

i
	​

)+
k=1
∑
m
	​

Ω(f
k
	​

)
	​


The first term measures prediction error.

The second term controls model complexity.

Step 4. Regularization

Each tree has a penalty

Ω(f)=γT+
2
1
	​

λ
j=1
∑
T
	​

w
j
2
	​

	​


where

T = number of leaves
w
j
	​

 = leaf weight
γ = penalty for adding leaves
λ = L2 regularization

This prevents overfitting.

Step 5. Taylor Approximation

Instead of minimizing the loss directly, XGBoost uses a second-order Taylor expansion.

For each sample,

g
i
	​

=
∂
y
^
	​

i
	​

∂L
	​


First derivative (Gradient)

and

h
i
	​

=
∂
y
^
	​

i
2
	​

∂
2
L
	​


Second derivative (Hessian)

Approximate loss:

L≈g
i
	​

f(x
i
	​

)+
2
1
	​

h
i
	​

f(x
i
	​

)
2
	​


Using both gradient and Hessian helps XGBoost converge faster and choose better splits.

Step 6. Best Leaf Weight

For each leaf,

w
∗
=−
H+λ
G
	​

	​


where

G=∑g
i
	​


and

H=∑h
i
	​

Step 7. Split Gain Formula

A split is accepted only if it increases the objective.

Gain=
2
1
	​

(
H
L
	​

+λ
G
L
2
	​

	​

+
H
R
	​

+λ
G
R
2
	​

	​

−
H+λ
G
2
	​

)−γ
	​


where

G
L
	​

,H
L
	​

 = Left child
G
R
	​

,H
R
	​

 = Right child
G,H = Parent node

The split with the highest positive gain is chosen.

Overall Workflow
Start Prediction
       │
       ▼
Compute Loss
       │
       ▼
Calculate Gradient (g)
       │
       ▼
Calculate Hessian (h)
       │
       ▼
Find Best Split Using Gain Formula
       │
       ▼
Build New Tree
       │
       ▼
Update Prediction
       │
       ▼
Repeat Until n_estimators Trees
Training XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    random_state=42
)

xgb.fit(X, y)
Why XGBoost Performs Better
Builds trees sequentially to correct previous errors.
Uses Gradient + Hessian (second-order optimization) for more accurate updates.
Includes L1/L2 regularization to reduce overfitting.
Automatically handles missing values.
Supports parallel computation for faster training.
Usually achieves higher accuracy than many other tree-based algorithms on structured/tabular data.

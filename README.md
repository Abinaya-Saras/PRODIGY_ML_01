# House Price Prediction

## Overview

This project predicts house prices using linear regression. It involves data cleaning, feature selection, and preprocessing.

## Dependencies

- numpy
- pandas
- seaborn
- tensorflow
- tensorflow_decision_forests
- scikit-learn

## Steps

1. **Data Loading**
   ```python
   import pandas as pd
   df_train = pd.read_csv('/content/train.csv')
   df_test = pd.read_csv('/content/test.csv')

2. **Data Cleaning**

Drop columns with >50% missing values.
Fill missing values for numerical columns with mean and categorical with mode.

df_cleaned = df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)

3.**Preprocessing**

**Feature Selection:** Use SelectKBest to select top 3 features.
**Normalization:** Scale features using MinMaxScaler.

from sklearn.feature_selection import **SelectKBest**, f_regression
from sklearn.preprocessing import **MinMaxScaler**

4. **Model Training**

Train a linear regression model and make predictions.

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

5. **Submission**

Save predictions to **submission.csv.**

output = pd.read_csv('/content/sample_submission.csv')
output['SalePrice'] = y_pred
output.to_csv('submission.csv', index=False)


This version covers the essential points while being concise.

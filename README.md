# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation:
Load the dataset, clean it (handle missing values or outliers), and split it into features (X) and target variable (y).
2.Train-Test Split:
Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.
3.Model Building:
Use the training data to build a linear regression model.
4.Model Evaluation:
Use metrics like R-squared and Mean Squared Error (MSE) to evaluate how well the model predicts car prices on the test set.


## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('car_data.csv')
data = data.dropna()
X = data.drop('Price', axis=1)  # Features
y = data['Price']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: 
RegisterNumber:  
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.

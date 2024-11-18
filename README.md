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
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: KUMAR G. 
RegisterNumber: 212223220048
*/
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Prasana v 
RegisterNumber: 212223040150
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
file_path = 'CarPrice.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]  # Features
y = df['price']  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# 1. Assumption: Linearity
plt.scatter(y_test, y_pred)
plt.title("Linearity: Observed vs Predicted Prices")
plt.xlabel("Observed Prices")
plt.ylabel("Predicted Prices")
plt.show()

# 2. Assumption: Independence (Durbin-Watson test)
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_test}")

# 3. Assumption: Homoscedasticity
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Homoscedasticity: Residuals vs Predicted Prices")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.show()

# 4. Assumption: Normality of residuals
sns.histplot(residuals, kde=True)
plt.title("Normality: Histogram of Residuals")
plt.show()

sm.qqplot(residuals, line='45')
plt.title("Normality: Q-Q Plot of Residuals")
plt.show()

# Insights
print("Check these outputs to verify assumptions for linear regression.")
```

## Output:
![image](https://github.com/user-attachments/assets/81a9963a-8d9a-46cc-b68c-1b7a81856c1c)
![image](https://github.com/user-attachments/assets/31e33a25-b6ff-41fb-bdd6-0054f9fb8bbe)
![image](https://github.com/user-attachments/assets/8fe89a94-f4af-4f84-917b-a983542df24e)
![image](https://github.com/user-attachments/assets/277a8aac-d764-49e2-ac72-5160e23337c6)
![image](https://github.com/user-attachments/assets/62257f5d-63e5-45dc-8ac0-8a8ab177e410)






## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.

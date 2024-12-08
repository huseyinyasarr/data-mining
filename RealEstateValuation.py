import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_excel("datasets/Real estate valuation data set.xlsx")

df = df.rename(columns={
    "No": "No",
    "X1 transaction date": "Transaction_Date",
    "X2 house age": "House_Age",
    "X3 distance to the nearest MRT station": "Distance_MRT",
    "X4 number of convenience stores": "Convenience_Stores",
    "X5 latitude": "Latitude",
    "X6 longitude": "Longitude",
    "Y house price of unit area": "Price"
})

# Remove the "No" column
df = df.drop(columns=["No"])

# missing values chechk
print("missing values:\n", df.isnull().sum())


X = df.drop("Price", axis=1)
y = df["Price"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train set / 5 = test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train set / 4 = validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# Creating models
lin_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=0.01)

# Training
lin_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)
lasso_reg.fit(X_train, y_train)

# Predictions on Test Set
y_pred_lin = lin_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)

# Performance Metrics (Test Set)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Result")
print("Linear Regression: MAE:", mae_lin, "MSE:", mse_lin, "R2:", r2_lin)
print("Ridge Regression:   MAE:", mae_ridge, "MSE:", mse_ridge, "R2:", r2_ridge)
print("Lasso Regression:   MAE:", mae_lasso, "MSE:", mse_lasso, "R2:", r2_lasso)




# The Relationship Between Distance and Price visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Distance_MRT'], y=df['Price'])
plt.title("The Relationship Between Distance and Price")
plt.xlabel("Distance to MRT Station")
plt.ylabel("Price")
plt.show()



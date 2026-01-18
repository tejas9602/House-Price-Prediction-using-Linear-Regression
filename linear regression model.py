import pandas as pd # type: ignore
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Load dataset safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "house_data.csv")

data = pd.read_csv(CSV_PATH)

# Select relevant features and target
features = ["GrLivArea", "BedroomAbvGr", "FullBath", "OverallQual"]
target = "SalePrice"

# Drop rows with missing values
data = data[features + [target]].dropna()

X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)



# 1. Actual vs Predicted Prices
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# 2. House Size vs Price
plt.figure()
plt.scatter(data["GrLivArea"], data["SalePrice"])
plt.xlabel("House Size (GrLivArea)")
plt.ylabel("Sale Price")
plt.title("House Size vs Sale Price")
plt.show()

# 3. Residual Plot (Errors)
residuals = y_test - predictions

plt.figure()
plt.scatter(predictions, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

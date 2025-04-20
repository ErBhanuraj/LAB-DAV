import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('Experience-Salary.csv')
data.columns = ['Experience', 'Salary']

# Reshape features and target
X = data[['Experience']]
y = data['Salary']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Model parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Predict salary for 2.3 years experience
X_new = pd.DataFrame({'Experience': [2.3]})
predicted_salary = model.predict(X_new)
print("Predicted Salary (2.3 yrs):", predicted_salary[0])

# Predict for min and max experience for regression line
X_line = pd.DataFrame({'Experience': [X.min()[0], X.max()[0]]})
y_line = model.predict(X_line)

# Scatter plot and regression line
plt.scatter(data['Experience'], data['Salary'], color='blue', label='Actual')
plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Evaluate model
y_pred = model.predict(X)
rmse = sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("RMSE:", rmse)
print("R-squared:", r2)

plt.show()

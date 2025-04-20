import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from mpl_toolkits.mplot3d import Axes3D

# Read the data
df = pd.read_csv("Student_Performance.csv")
df.head(5)

# Feature selection
col = "Performance Index"
x = df.drop(["Extracurricular Activities", "Performance Index"], axis=1)
y = df["Performance Index"]

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, shuffle=True)

# Apply SelectKBest with chi2
select_k_best = SelectKBest(score_func=chi2, k=2)
X_train_k_best = select_k_best.fit_transform(xtrain, ytrain)
print("Selected features:", xtrain.columns[select_k_best.get_support()])

# Linear regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Predictions
ypred = model.predict(xtest)

# 3D Plot visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Using Hours Studied, Sample Question Papers Practiced, and actual Performance Index for 3D plot
ax.scatter(xtest["Hours Studied"], xtest["Sample Question Papers Practiced"], ytest, color='green')
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Sample Question Papers Practiced')
ax.set_zlabel('Performance Index')
ax.set_title('Multiple Regression Plot')
plt.show()

# Evaluate the model
print("R-squared:", r2_score(ytest, ypred))
print("Mean Squared Error:", mean_squared_error(ytest, ypred))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
import os
os.chdir(r"E:\New folder\ML\StartUp")
startups = pd.read_csv("StartUp.csv")
df = startups.copy()

# Check for null values
print(df.isna().sum())

# Drop non-numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Calculate correlation and draw graphs
core = df_numeric.corr()
sns.heatmap(core, annot=True)
sns.scatterplot(x="R&D Spend", y="Profit", data=df, color="red")
df.hist(figsize=(10,10))

# Get Statistical Info about data
print(df.describe().T)

# Encode non-numeric categorical variable 'State'
label_encoder = LabelEncoder()
df['State'] = label_encoder.fit_transform(df['State'])

# Define X and y
x = df.drop("Profit", axis=1)
y = df["Profit"]

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=35)

# Fit a Linear Regression model using scikit-learn
lm = LinearRegression()
model = lm.fit(xTrain, yTrain)
y_pred = model.predict(xTest)

# Calculate Mean Absolute Error (MAE)
MAE = mean_absolute_error(yTest, y_pred)

# Print MAE
print("Mean Absolute Error:", MAE)

# Print bar char
df1 = startups.head(10)
df1.plot(kind="bar", figsize=(10,10))
plt.grid(which="major",linestyle="-", linewidth="0.5",color="green")

# Print model intercept and coefficients
print('Intercept of the model:\n', lm.intercept_)
print("="*50)
print('Coefficient of the line:\n', lm.coef_)

# Fit OLS model with statsmodels
# First, fit the model without adding a constant term for intercept
model = sm.OLS(y.astype(float), x.astype(float)).fit()
print(model.summary())

# Then, add a constant term for intercept
x = sm.add_constant(x)
model = sm.OLS(y.astype(float), x.astype(float)).fit()
print(model.summary())

# Drop 'Administration', 'Marketing Spend', and 'State' columns
x = x.drop(['Administration', 'Marketing Spend', 'State'], axis=1)
model = sm.OLS(y.astype(float), x.astype(float)).fit()
print(model.summary())
plt.show()

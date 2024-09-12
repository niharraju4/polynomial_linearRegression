## Table of Contents
1. [Load Libraries](#load-libraries)
2. [Generate Data](#generate-data)
3. [Simple Linear Regression](#simple-linear-regression)
4. [Polynomial Regression](#polynomial-regression)
5. [Prediction of New Data](#prediction-of-new-data)
6. [Pipeline Concepts](#pipeline-concepts)
7. [Results](#results)
8. [Summary](#summary)
9. [Additional Steps and Suggestions](#additional-steps-and-suggestions)

---
## Author: Nihar Raju
## Load Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
```

---

## Generate Data
```python
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + 1.5 * X + 2 + np.random.randn(100, 1)

plt.scatter(X, y, color='g')
plt.xlabel('X dataset')
plt.ylabel('Y dataset')
plt.show()
```

---

## Simple Linear Regression
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_1 = LinearRegression()
regression_1.fit(X_train, y_train)

score = r2_score(y_test, regression_1.predict(X_test))
print("R2 Score (Simple Linear Regression):", score)

plt.plot(X_train, regression_1.predict(X_train), color='r')
plt.scatter(X_train, y_train)
plt.xlabel("X Dataset")
plt.ylabel("Y")
plt.show()
```

---

## Polynomial Regression
### Degree 2
```python
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)

score = r2_score(y_test, y_pred)
print("R2 Score (Polynomial Regression Degree 2):", score)
print("Coefficients:", regression.coef_)
print("Intercept:", regression.intercept_)

plt.scatter(X_train, regression.predict(X_train_poly))
plt.scatter(X_train, y_train)
plt.show()
```

### Degree 3
```python
poly = PolynomialFeatures(degree=3, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)

score = r2_score(y_test, y_pred)
print("R2 Score (Polynomial Regression Degree 3):", score)
```

---

## Prediction of New Data
```python
X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = regression.predict(X_new_poly)

plt.plot(X_new, y_new, "r-", linewidth=2, label="New Predictions")
plt.plot(X_train, y_train, "b.", label='Training points')
plt.plot(X_test, y_test, "g.", label='Testing points')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

---

## Pipeline Concepts
```python
def poly_regression(degree):
    X_new = np.linspace(-3, 3, 200).reshape(200, 1)

    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    lin_reg = LinearRegression()
    poly_regression = Pipeline([
        ("poly_features", poly_features),
        ("lin_reg", lin_reg)
    ])
    poly_regression.fit(X_train, y_train)
    y_pred_new = poly_regression.predict(X_new)

    plt.plot(X_new, y_pred_new, 'r', label="Degree " + str(degree), linewidth=2)
    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.plot(X_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-4, 4, 0, 10])
    plt.show()

poly_regression(6)
```

---

## Results

### Data Generation
- **Scatter Plot of Generated Data:**
  ```python
  plt.scatter(X, y, color='g')
  plt.xlabel('X dataset')
  plt.ylabel('Y dataset')
  plt.show()
  ```

### Simple Linear Regression
- **R2 Score:**
  ```python
  print("R2 Score (Simple Linear Regression):", score)
  ```
- **Visualization:**
  ```python
  plt.plot(X_train, regression_1.predict(X_train), color='r')
  plt.scatter(X_train, y_train)
  plt.xlabel("X Dataset")
  plt.ylabel("Y")
  plt.show()
  ```

### Polynomial Regression (Degree 2)
- **R2 Score:**
  ```python
  print("R2 Score (Polynomial Regression Degree 2):", score)
  ```
- **Coefficients and Intercept:**
  ```python
  print("Coefficients:", regression.coef_)
  print("Intercept:", regression.intercept_)
  ```
- **Visualization:**
  ```python
  plt.scatter(X_train, regression.predict(X_train_poly))
  plt.scatter(X_train, y_train)
  plt.show()
  ```

### Polynomial Regression (Degree 3)
- **R2 Score:**
  ```python
  print("R2 Score (Polynomial Regression Degree 3):", score)
  ```

### Prediction of New Data
- **Visualization:**
  ```python
  plt.plot(X_new, y_new, "r-", linewidth=2, label="New Predictions")
  plt.plot(X_train, y_train, "b.", label='Training points')
  plt.plot(X_test, y_test, "g.", label='Testing points')
  plt.xlabel("X")
  plt.ylabel("y")
  plt.legend()
  plt.show()
  ```

### Pipeline Concepts
- **Visualization for Degree 6:**
  ```python
  poly_regression(6)
  ```

---

## Summary

I have performed a comprehensive analysis of polynomial regression, including data generation, simple linear regression, polynomial regression with different degrees, prediction of new data, and the use of pipelines. These steps provide a solid foundation for understanding and applying polynomial regression. Keep exploring and visualizing the data to uncover more insights!

---

## Additional Steps and Suggestions

- **Hyperparameter Tuning:**
  Experiment with different degrees of polynomial features to find the optimal model.

- **Cross-Validation:**
  Use cross-validation to evaluate the performance of your models more robustly.

- **Regularization:**
  Consider adding regularization (e.g., Ridge or Lasso regression) to prevent overfitting, especially with higher-degree polynomials.

- **Feature Engineering:**
  Explore additional features or transformations that might improve the model's performance.

- **Model Interpretation:**
  Analyze the coefficients of the polynomial regression to understand the relationship between the features and the target variable.

Here's an example of how you might perform cross-validation:

```python
from sklearn.model_selection import cross_val_score

poly_features = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
scores = cross_val_score(lin_reg, X_poly, y, cv=5, scoring='r2')
print("Cross-Validation R2 Scores:", scores)
print("Mean R2 Score:", scores.mean())
```

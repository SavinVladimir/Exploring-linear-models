import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

import warnings
warnings.filterwarnings("ignore")

# Подготовка данных

data = pd.read_csv('data/Housing_Prices.csv')

X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
          'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pl = Pipeline([('std_scalar', StandardScaler())])

X_train = pl.fit_transform(X_train)
X_test = pl.transform(X_test)


def add_metrics_to_dataframe(true_value, predicted_value):
    mae = metrics.mean_absolute_error(true_value, predicted_value)
    mse = metrics.mean_squared_error(true_value, predicted_value)
    rmse = np.sqrt(metrics.mean_squared_error(true_value, predicted_value))
    # msle = metrics.mean_squared_log_error(true_value, predicted_value)
    r2 = metrics.r2_score(true_value, predicted_value)

    return mae, mse, rmse, r2

# Множественная регрессия

linear_regression = LinearRegression(normalize=True)
linear_regression.fit(X_train, y_train)

test_pred_linear_regression = linear_regression.predict(X_test)

# Робастная регрессия

ransac_regression = RANSACRegressor(base_estimator=LinearRegression(), max_trials=100)
ransac_regression.fit(X_train, y_train)

test_pred_ransac_regression = ransac_regression.predict(X_test)

# Регрессия гребня

ridge_regression = Ridge(alpha=5, solver='cholesky', tol=0.001, random_state=42)
ridge_regression.fit(X_train, y_train)

test_pred_ridge_regression = ridge_regression.predict(X_test)

# Регрессия LASSO

LASSO_regression = Lasso(alpha=0.1, precompute=True, positive=True, selection='random', random_state=42)
LASSO_regression.fit(X_train, y_train)

test_pred_LASSO_regression = ridge_regression.predict(X_test)

# Полиномиальная регрессия

polynomial_regression = PolynomialFeatures(degree=2)

X_train_2_d = polynomial_regression.fit_transform(X_train)
X_test_2_d = polynomial_regression.transform(X_test)

linear_regression_for_polynomial = LinearRegression(normalize=True)
linear_regression_for_polynomial.fit(X_train_2_d, y_train)

test_pred_polynomial_regression = linear_regression_for_polynomial.predict(X_test_2_d)


# Стохастический Градиентный спуск

sgd_regression = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
sgd_regression.fit(X_train, y_train)

test_pred_sgd_regression = sgd_regression.predict(X_test)

# Метод опорных векторов

svm_regression = SVR(kernel='rbf', C=1000000, epsilon=0.001)
svm_regression.fit(X_train, y_train)

test_pred_svm_regression = svm_regression.predict(X_test)

# Формирование сводной таблицы с результатами

results_of_models = pd.DataFrame(data=[["Множественная регрессия",
                                        *add_metrics_to_dataframe(y_test, test_pred_linear_regression)],
                                       ["Робастная регрессия",
                                        *add_metrics_to_dataframe(y_test, test_pred_ransac_regression)],
                                       ["Гребневая регрессия",
                                        *add_metrics_to_dataframe(y_test, test_pred_ridge_regression)],
                                       ["Регрессия LASSO",
                                        *add_metrics_to_dataframe(y_test, test_pred_LASSO_regression)],
                                       ["Полиномиальная регрессия",
                                        *add_metrics_to_dataframe(y_test, test_pred_polynomial_regression)],
                                       ["Стохастический градиентный спуск",
                                        *add_metrics_to_dataframe(y_test, test_pred_sgd_regression)],
                                       ["Метод опорных векторов",
                                        *add_metrics_to_dataframe(y_test, test_pred_svm_regression)],
                                       ],
                                 columns=['Алгоритм', 'MAE', 'MSE', 'RMSE', 'R2 Square'])

results_of_models.to_csv('The_results_of_the_algorithms.csv', index=False)

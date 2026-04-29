import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.ensemble import VotingRegressor, BaggingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('processed_variant_13.csv')
X = df.drop(columns=['BMI'])
y = df['BMI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Size of training sample: {X_train.shape}")
print(f"Size of test sample: {X_test.shape}")

models_and_params = {
    'Linear Regression': (LinearRegression(), {}),
    'KNeighbors Regressor': (KNeighborsRegressor(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }),
    'Decision Tree Regressor': (DecisionTreeRegressor(random_state=42), {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }),
    'SVR': (SVR(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    'Ridge': (Ridge(), {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky']
    }),
    'Lasso': (Lasso(), {
        'alpha': [0.01, 0.1, 1.0]
    }),
    'ElasticNet': (ElasticNet(), {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.2, 0.5, 0.8]
    }),
    'Random Forest Regressor': (RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }),
    'AdaBoost Regressor': (AdaBoostRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 1.0]
    }),
    'Gradient Boosting Regressor': (GradientBoostingRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    }),
    'XGBoost': (XGBRegressor(random_state=42, objective='reg:squarederror'), {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }),
    'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    })
}

best_estimators = {}
results = []

print("Learning individual models...")
for name, (model, params) in models_and_params.items():
    if params:
        search = RandomizedSearchCV(estimator=model, param_distributions=params,
                                    n_iter=5, cv=3, scoring='neg_mean_absolute_error',
                                    random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params_str = str(search.best_params_)
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params_str = "Default"

    best_estimators[name] = best_model
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({'Model': name, 'MAE': mae, 'MSE': mse, 'R2': r2, 'Best parameters': best_params_str})
    print(f"{name} learned.")

print("\nLearning ensemble models...")

voting_reg = VotingRegressor(estimators=[
    ('lr', best_estimators['Linear Regression']),
    ('rf', best_estimators['Random Forest Regressor']),
    ('gb', best_estimators['Gradient Boosting Regressor'])
])
voting_reg.fit(X_train, y_train)
best_estimators['Voting Ensemble'] = voting_reg
print("Voting Ensemble learned.")

bagging_reg = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                               n_estimators=50, random_state=42)
bagging_reg.fit(X_train, y_train)
best_estimators['Bagging Ensemble'] = bagging_reg
print("Bagging Ensemble learned.")

stacking_reg = StackingRegressor(
    estimators=[
        ('ridge', best_estimators['Ridge']),
        ('rf', best_estimators['Random Forest Regressor']),
        ('svr', best_estimators['SVR'])
    ],
    final_estimator=LinearRegression()
)
stacking_reg.fit(X_train, y_train)
best_estimators['Stacking Ensemble'] = stacking_reg
print("Stacking Ensemble learned.")

for name in ['Voting Ensemble', 'Bagging Ensemble', 'Stacking Ensemble']:
    model = best_estimators[name]
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': mae, 'MSE': mse, 'R2': r2, 'Best parameters': 'Ensemble'})

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\n Final results for all models:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = best_estimators[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(14, 6))

for index, row in results_df.iterrows():
    model_name = row['Model']
    r2_val = row['R2']

    model = best_estimators[model_name]
    y_pred = model.predict(X_test)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Factual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Correlations chart ({model_name})\n$R^2$ = {r2_val:.4f}")

    residuals = y_test - y_pred
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Chart of residuals ({model_name})")

    plt.tight_layout()
    plt.show()

    # Try to take correlations coefficient for each model
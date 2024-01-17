import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

class RecentDataKeeper:
    def __init__(self, max_lag):
        self.max_lag = max_lag
        self.recent_data = pd.DataFrame()

    def update(self, new_data):
        self.recent_data = pd.concat([self.recent_data, new_data], ignore_index=True)[-self.max_lag:]

    def create_group_lag_features(self, df, group_column, lags, time_column):
        combined_data = pd.concat([self.recent_data, df]).sort_values(by=[group_column, time_column])
        for lag in lags:
            combined_data[f'lag_{lag}'] = combined_data.groupby(group_column)['y'].shift(lag)
        return combined_data.dropna()

class ModelOptimizer:
    def __init__(self, n_trials=100, max_lag=30):
        self.n_trials = n_trials
        self.keeper = RecentDataKeeper(max_lag)

    def objective(self, trial, X_train, y_train):
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'objective': 'reg:squarederror'
        }
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, early_stopping_rounds=10, verbose=False)
        preds = model.predict(X_train)
        return mean_squared_error(y_train, preds, squared=False)

    def fit(self, X, y, group_column, lags, time_column):
        self.keeper.update(X)
        X_with_lags = self.keeper.create_group_lag_features(X, group_column, lags, time_column)
        y = y[X_with_lags.index]

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X_with_lags):
            X_train, X_test = X_with_lags.iloc[train_index], X_with_lags.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=self.n_trials)

            best_params = study.best_params
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train, early_stopping_rounds=10, verbose=False)
            preds = model.predict(X_test)
            print(f"Fold RMSE: {rmse}")

        return model

# 모델 학습 및 테스트 데이터 예측
optimizer = ModelOptimizer(n_trials=100, max_lag=30)
lags = [1, 2, ...]  # 사용할 라그 설정
group_column = 'MATR'
time_column = 'time_column'  # 시간 컬럼명
optimizer.fit(X_train, y_train, group_column, lags, time_column)

new_data = pd.read_csv('new_test_data.csv')
optimizer.keeper.update(new_data)
X_test_with_lags = optimizer.keeper.create_group_lag_features(new_data, group_column, lags, time_column)

predictions = model.predict(X_test_with_lags)

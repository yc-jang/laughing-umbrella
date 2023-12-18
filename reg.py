import numpy as np
import optuna
from pytorch_tabnet.tab_model import TabNetRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

class TabNetXGBoostRegressor:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.tabnet_params = None
        self.xgb_params = None
        self.model_tabnet = None
        self.model_xgb = None
        self.best_trial = None
        self.weight_tabnet = 0.5  # 초기 가중치

    def objective(self, trial, X_train, y_train, X_valid, y_valid):
        # TabNet 하이퍼파라미터
        tabnet_params = {
            'n_d': trial.suggest_int('n_d', 16, 64),
            'n_a': trial.suggest_int('n_a', 16, 64),
            'n_steps': trial.suggest_int('n_steps', 1, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'n_independent': trial.suggest_int('n_independent', 1, 5),
            'n_shared': trial.suggest_int('n_shared', 1, 5),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
            'optimizer_params': {'lr': 2e-2, 'weight_decay': 1e-5},
            'max_epochs': 100,
            'patience': 10,
            'batch_size': 256,
            'virtual_batch_size': 128,
            'num_workers': 0
        }

        # XGBoost 하이퍼파라미터
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'objective': 'reg:squarederror'
        }

        # 모델 학습
        model_tabnet = TabNetRegressor(**tabnet_params)
        model_tabnet.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)])

        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10, verbose=False)

        # 예측 및 평가
        preds_tabnet = model_tabnet.predict(X_valid.values)
        preds_xgb = model_xgb.predict(X_valid)
        final_preds = self.combine_predictions(preds_tabnet, preds_xgb)

        rmse = mean_squared_error(y_valid, final_preds, squared=False)
        return rmse

    
    def combine_predictions(self, preds_tabnet, preds_xgb):
        weight_xgb = 1 - self.weight_tabnet
        final_preds = self.weight_tabnet * preds_tabnet + weight_xgb * preds_xgb
        return final_preds

    def fit(self, X_train, y_train, X_valid, y_valid):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_valid, y_valid), n_trials=self.n_trials)

        self.best_trial = study.best_trial
        self.tabnet_params = study.best_trial.params
        self.xgb_params = study.best_trial.params
        self.weight_tabnet = study.best_trial.params.get('weight_tabnet', 0.5)

        self.model_tabnet = TabNetRegressor(**self.tabnet_params)
        self.model_tabnet.fit(X_train.values, y_train.values)

        self.model_xgb = xgb.XGBRegressor(**self.xgb_params)
        self.model_xgb.fit(X_train, y_train)

    def predict(self, X):
        preds_tabnet = self.model_tabnet.predict(X.values)
        preds_xgb = self.model_xgb.predict(X)
        return self.combine_predictions(preds_tabnet, preds_xgb)
    
    
    def visualize_tabnet_masks(self, X, num_features=10):
        """
        TabNet 모델의 피처 중요도 마스크를 Plotly를 사용하여 시각화합니다.
        :param X: 입력 데이터
        :param num_features: 시각화할 상위 피처의 수 (기본값 10)
        """
        mask, _ = self.model_tabnet.explain(X.values)
        for idx, feature in enumerate(X.columns[:num_features]):
            fig = go.Figure(data=go.Heatmap(
                z=mask[:, idx].reshape(-1, 1),
                colorscale='Viridis'
            ))
            fig.update_layout(
                title=f"Feature Mask for {feature}",
                yaxis=dict(title='Sample'),
                xaxis=dict(title='Importance')
            )
            fig.show()

    def visualize_shap_values(self, X, model, num_features=10):
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, plot_type="bar", max_display=num_features)

    def visualize_training_curves(self):
        if hasattr(self.model_tabnet, 'history'):
            fig = go.Figure()
            for key in self.model_tabnet.history.keys():
                fig.add_trace(go.Scatter(x=list(range(len(self.model_tabnet.history[key]))), y=self.model_tabnet.history[key], mode='lines', name=key))
            fig.update_layout(title='TabNet Training Curves', xaxis_title='Epoch', yaxis_title='Metric')
            fig.show()

        if hasattr(self.model_xgb, 'evals_result'):
            results = self.model_xgb.evals_result()
            for eval_set in results.keys():
                for metric in results[eval_set].keys():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(results[eval_set][metric]))), y=results[eval_set][metric], mode='lines', name=f'{eval_set}-{metric}'))
                    fig.update_layout(title=f'XGBoost Training Curves - {eval_set}', xaxis_title='Epoch', yaxis_title=metric)
                    fig.show()
                    
    def evaluate_model(self, X_test, y_test):
        preds_tabnet = self.model_tabnet.predict(X_test.values)
        preds_xgb = self.model_xgb.predict(X_test)
        final_preds = 0.5 * preds_tabnet + 0.5 * preds_xgb

        rmse = mean_squared_error(y_test, final_preds, squared=False)
        return rmse


from sklearn.datasets import make_regression

# 임의의 데이터셋 생성 (여기서는 make_regression 함수를 사용합니다)
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y, name='target')

# 데이터셋을 train/test/validation으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.2 * 0.8 = 0.16

# 모델 초기화 및 학습
model = TabNetXGBoostRegressor(n_trials=10)
model.fit(X_train, y_train, X_valid, y_valid)

# 예측
predictions = model.predict(X_test)

# 모델 평가
model_performance = model.evaluate_model(X_test, y_test)
print(f"Model Performance (RMSE): {model_performance}")

# 시각화: TabNet 피처 중요도 마스크
model.visualize_tabnet_masks(X_test, num_features=5)

# 시각화: SHAP 값
model.visualize_shap_values(X_test, model.model_xgb, num_features=5)

# 시각화: 학습 곡선
model.visualize_training_curves()

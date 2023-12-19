import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)

def plot_metrics(metrics: dict):
    labels = list(metrics.keys())
    rmse_values = [m['rmse'] for m in metrics.values()]
    r2_values = [m['r2'] for m in metrics.values()]

    fig = go.Figure(data=[
        go.Bar(name='RMSE', x=labels, y=rmse_values),
        go.Bar(name='R² Score', x=labels, y=r2_values)
    ])
    
    # Update layout
    fig.update_layout(barmode='group', title='Model Performance Comparison',
                      xaxis_title='Models', yaxis_title='Values',
                      legend_title='Metrics')
    fig.show()

# 예시 사용법
y_true = np.array([실제값 리스트])
y_pred_model1 = np.array([모델1 예측값 리스트])
y_pred_model2 = np.array([모델2 예측값 리스트])

metrics = {
    'Model 1': {
        'rmse': calculate_rmse(y_true, y_pred_model1),
        'r2': calculate_r2(y_true, y_pred_model1)
    },
    'Model 2': {
        'rmse': calculate_rmse(y_true, y_pred_model2),
        'r2': calculate_r2(y_true, y_pred_model2)
    }
}

plot_metrics(metrics)

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


####################################################
# 'dtm' 컬럼을 datetime 객체로 변환
df['dtm'] = pd.to_datetime(df['dtm'])

# 특정 날짜 설정 (예시로 '2023-01-01' 사용)
cutoff_date = pd.to_datetime('2023-01-01')

# 날짜 기준으로 데이터 분리
before_cutoff = df[df['dtm'] < cutoff_date]
after_cutoff = df[df['dtm'] >= cutoff_date]

# 시각화
fig = go.Figure()

# 특정 날짜 이전 데이터
fig.add_trace(go.Scatter(x=before_cutoff['dtm'], y=before_cutoff['y'],
                         mode='lines', name='Before Cutoff'))

# 특정 날짜 이후 데이터 (빨간색으로 표시)
fig.add_trace(go.Scatter(x=after_cutoff['dtm'], y=after_cutoff['y'],
                         mode='lines', name='After Cutoff', line=dict(color='red')))

# 그래프 레이아웃 설정
fig.update_layout(title='Y Data with Cutoff Date', xaxis_title='Date', yaxis_title='Y Value')

# 그래프 보이기
fig.show()

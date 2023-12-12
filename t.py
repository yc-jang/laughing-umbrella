#%%
import pandas as pd
import numpy as np

def prepare_data():
    """
    샘플 데이터 생성 및 초기 준비
    """
    np.random.seed(0)
    s_dates = pd.date_range('2023-01-01', periods=90, freq='D')
    s = pd.DataFrame({
        '교체일': np.random.choice(s_dates, size=20),
        '단수': np.random.randint(1, 4, size=20),
        'decay': 1,
    }).drop_duplicates('교체일').sort_values('교체일')

    m_dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    m = pd.DataFrame({'date': m_dates})

    return s, m

def process_sheet_data_updated(df):
    df = df.drop(columns=['라인'])

    # '호기'와 '상/하' 열의 NaN 값을 앞의 값으로 채움
    df['호기'] = df['호기'].fillna(method='ffill')
    df['상/하'] = df['상/하'].fillna(method='ffill')

    # '수량'과 '교체일'이 있는 첫 번째 행 제거
    df = df.drop(0)

    reshaped_data = pd.DataFrame()

    # 1월부터 12월까지 각 월에 대해 처리
    for month in range(1, 13):
        month_str = str(month) + '월'
        quantity_column = month_str
        replacement_date_column = 'Unnamed: ' + str(month * 2 + 2)  # '교체일' 데이터가 있는 열

        month_data = pd.DataFrame({
            '호기': df['호기'],
            '상/하': df['상/하'],
            '수량': df[quantity_column],
            '교체일': df[replacement_date_column]
        })

        # 빈 값이 아닌 행만 선택
        month_data = month_data.dropna(subset=['수량', '교체일'])

        
        reshaped_data = pd.concat([reshaped_data, month_data], ignore_index=True)
        

    reshaped_data['호기'] = reshaped_data['호기'].astype(str)
    reshaped_data['상/하'] = reshaped_data['상/하'].astype(str)
    reshaped_data['수량'] = reshaped_data['수량'].astype(int)
    reshaped_data['교체일'] = pd.to_datetime(reshaped_data['교체일'], format='%Y-%m-%d', errors='coerce')

    return reshaped_data


def calculate_decay_values_corrected(s, m):
    """
    각 단수별로 decay 값을 계산하되, 교체일에 decay 값으로 초기화하고 감소되는 로직 적용
    교체일 이전 날짜에 대해서도 첫 교체일의 decay 값을 사용
    """
    start_date = m['date'].min()
    end_date = m['date'].max()
    result = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='D')})

    for stage in range(1, 4):
        current_stage = s[s['단수'] == stage][['교체일', 'decay']].reset_index(drop=True)
        temp_df = pd.DataFrame({
            'date': pd.date_range(start_date, end_date, freq='D'),
            'value': np.nan,
            '단수': stage
        })

        # 첫 교체일 이전 날짜에 첫 교체일의 decay 값을 사용
        if not current_stage.empty:
            first_replace_date = current_stage.iloc[0]['교체일']
            first_decay = current_stage.iloc[0]['decay']
            temp_df.loc[temp_df['date'] < first_replace_date, 'value'] = first_decay

        # 교체일에 decay 값으로 초기화하고, 이후 감소 로직 적용
        for i in range(len(current_stage)):
            replace_date = current_stage.at[i, '교체일']
            decay = current_stage.at[i, 'decay']
            temp_df.loc[temp_df['date'] == replace_date, 'value'] = decay

            if i < len(current_stage) - 1:
                next_replace_date = current_stage.at[i + 1, '교체일']
                date_range = temp_df[(temp_df['date'] > replace_date) & (temp_df['date'] < next_replace_date)].index
            else:
                date_range = temp_df[temp_df['date'] > replace_date].index

            days_since_replace = (temp_df.loc[date_range, 'date'] - replace_date).dt.days
            temp_df.loc[date_range, 'value'] = decay * np.exp(-0.1 * days_since_replace)

        # 이전 교체일의 decay 값으로 초기화
        temp_df['value'].fillna(method='ffill', inplace=True)

        result = pd.merge(result, temp_df[['date', 'value']], on='date', how='left').rename(columns={'value': f'value_stage_{stage}'})

    return result

file_path = 'saggar.xlsx'
excel_data = pd.ExcelFile(file_path)

processed_sheets_updated = []
for sheet_name in excel_data.sheet_names:
    df = pd.read_excel(excel_data, sheet_name=sheet_name)
    processed_data = process_sheet_data_updated(df)
    processed_sheets_updated.append(processed_data)

merged = pd.concat(processed_sheets_updated, ignore_index=True)
layer_mapping = {
    '상단(80mm)': 3,
    '중단(50mm)': 2,
    '하단(40mm)': 1
}
merged['단'] = merged['상/하'].map(layer_mapping)

#%%
# 데이터 준비
s, m = prepare_data()

# Decay 값 계산
result = calculate_decay_values_corrected(s, m)
# %%
import plotly.express as px

# Plotly를 사용하여 그래프 그리기
fig = px.line(result, x='date', y=[f'value_stage_{stage}' for stage in range(1, 4)],
              labels={'value': 'Decay Value', 'date': 'Date'},
              title='Decay Values Over Time for Each Stage')

# 각 단수별 교체일에 대한 표시 추가
for stage in range(1, 4):
    for _, row in s[s['단수'] == stage].iterrows():
        fig.add_scatter(x=[row['교체일']], y=[row['decay']], mode='markers', marker=dict(color='red', size=10),
                        showlegend=False)

fig.show()
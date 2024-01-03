#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import plotly.express as px
import plotly.graph_objects as go
import umap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import statsmodels.api as sm
from tqdm import tqdm

#%%
class EDA:
    def __init__(self, df, target_column, drop_columns=None, mini=False, sample_fraction=0.1):
        self.df = df.copy()
        self.target_column = target_column
        self.mini = mini

        # Drop specified columns if any
        if drop_columns:
            self.df.drop(drop_columns, axis=1, inplace=True)

        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # 샘플링 옵션
        if self.mini:
            self.df = self.df.sample(frac=sample_fraction, random_state=42)


    def display_basic_info(self):
        print("Basic Information of the Dataset:")
        print("Shape:", self.df.shape)
        print("\nFirst 5 Rows:\n", self.df.head())
        print("\nData Types:\n", self.df.dtypes)
        print("\nSummary Statistics:\n", self.df.describe())

    def plot_correlation_matrix(self):
        """
        데이터셋 내의 모든 변수들 간의 상관관계를 히트맵 형태로 시각화합니다.
        상관계수는 -1부터 +1까지의 값을 가지며, 색상의 강도가 이 값을 나타냅니다.
        밝은 색은 강한 양의 상관관계를, 어두운 색은 강한 음의 상관관계를 나타냅니다.
        0에 가까울수록 색상은 옅어지며, 이는 변수 간 관계가 약하거나 없음을 의미합니다.

        이 행렬은 다음과 같은 분석에 유용합니다:
        1. 다중공선성 문제 식별: 두 변수 간의 상관계수가 매우 높으면(예: 0.8 이상) 
        이 변수들은 서로 강하게 연관되어 있으며, 이는 회귀 모델에서 다중공선성 문제를 일으킬 수 있습니다.
        2. 중요한 특성 선별: 타겟 변수와 높은 상관관계를 보이는 변수들은 모델에서 중요한 역할을 할 수 있으며,
        이러한 변수들은 모델의 예측 성능 향상에 기여할 수 있습니다.

        Returns:
            None: 상관관계 행렬을 히트맵 형태로 시각화합니다.
        """
        corr_matrix = self.df.corr()
        fig = px.imshow(corr_matrix, title="Correlation Matrix")
        fig.show()

    def perform_rfe(self, n_features=5, force=False):
        """
        재귀적 특성 제거(RFE)를 사용하여 모델의 가장 중요한 특성들을 식별합니다.
        RFE는 모델을 여러 번 피팅하면서 가장 중요도가 낮은 특성을 한 번에 하나씩 제거합니다.
        이 과정은 지정된 특성의 수에 도달할 때까지 반복됩니다.

        이 방법은 특히 높은 차원의 데이터셋에서 유용하며, 모델의 성능에 가장 큰 영향을 미치는
        특성들을 선별하는 데 도움을 줍니다. 

        Args:
            n_features (int): 최종적으로 선택할 특성의 수입니다.
            force (bool): 기존에 계산된 RFE 결과를 무시하고 새로 계산할지 여부.

        Returns:
            pd.Series: 선택된 특성들의 boolean 마스크. True 값은 선택된 특성을 나타냅니다.

        이 함수의 결과는 plot_rfe_results 함수에서 시각화될 수 있으며, 
        어떤 특성들이 모델에 중요한 영향을 미치는지를 바탕으로 효과적인 특성 선택이 가능합니다.
        """
        if not hasattr(self, 'rfe_features') or force:
            X = self.df.drop(self.target_column, axis=1)
            y = self.df[self.target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression()
            rfe = RFE(model, n_features_to_select=n_features)
            rfe.fit(X_train, y_train)
            self.rfe_features = pd.Series(rfe.support_, index=X.columns)
        return self.rfe_features

    def plot_rfe_results(self, n_features=5):
        self.perform_rfe(n_features)
        fig = px.bar(self.rfe_features, title="Feature Importance (RFE)")
        fig.show()

    def perform_pca_2d(self, n_components=2, force=False):
        """
        2차원 주성분 분석(PCA)을 수행하여 고차원 데이터를 2차원으로 축소합니다.
        이 방법은 데이터에서 가장 중요한 변동성을 포착하고, 이를 두 개의 주성분으로 요약합니다.
        
        2차원 PCA는 데이터의 핵심 구조를 시각적으로 이해하는 데 유용하며,
        특히 고차원 데이터셋의 패턴과 관계를 간결하게 표현할 수 있습니다.
        
        Args:
        n_components (int): 축소할 차원의 수, 여기서는 2.
        force (bool): 기존에 계산된 PCA 결과를 무시하고 새로 계산할지 여부.

        """
        if not hasattr(self, 'pca_df') or force:
            X = self.df.drop(self.target_column, axis=1)
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X)

            # PCA 결과 DataFrame 생성 및 인덱스 맞추기
            self.pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
            self.pca_df.index = self.df.index  # 원본 DataFrame의 인덱스를 pca_df에 복사
            self.pca_df['target'] = self.df[self.target_column]

    def plot_pca_2d_results(self, n_components=2):
        self.perform_pca_2d(n_components)
        fig = px.scatter(self.pca_df, x='PC1', y='PC2', color='target', title="PCA (2 components)")
        fig.show()
    
    def perform_pca_3d(self, n_components=3, force=False):
        """
        3차원 PCA를 수행하는 함수입니다.
        고차원 데이터를 3차원으로 축소하여 더욱 복잡한 구조와 패턴을 탐색할 수 있습니다.
        3차원 공간에서 데이터의 변동성을 시각화하여 더욱 입체적인 데이터 분석을 가능하게 합니다.
        """
        if not hasattr(self, 'pca_df_3d') or force:
            X = self.df.drop(self.target_column, axis=1)
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X)

            # PCA 결과 DataFrame 생성 및 인덱스 맞추기
            self.pca_df_3d = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
            self.pca_df_3d.index = self.df.index  # 원본 DataFrame의 인덱스를 pca_df_3d에 복사
            self.pca_df_3d['target'] = self.df[self.target_column]

    def plot_pca_3d_results(self, n_components=3):
        self.perform_pca_3d(n_components)
        fig = px.scatter_3d(self.pca_df_3d, x='PC1', y='PC2', z='PC3', color='target', title="PCA (3 components)")
        fig.show()

    def perform_pls(self, n_components=2, force=False):
        """
        부분 최소 제곱 회귀(PLS)를 사용하여 데이터를 분석합니다.
        PLS는 응답 변수와 예측 변수 간의 관계를 모델링하며, 고차원 데이터에서 유용합니다.
        PLS로 차원이 축소된 데이터를 산점도로 나타냅니다.
        이를 통해 데이터의 패턴과 구조를 시각적으로 이해할 수 있습니다.

        Args:
            n_components (int): 사용할 구성 요소의 수입니다.
            force (bool): 기존에 계산된 PLS 결과를 무시하고 새로 계산할지 여부입니다.

        Returns:
            None: 계산된 PLS 결과는 클래스의 내부 상태에 저장합니다.
        """
        if not hasattr(self, 'pls_scores') or force:
            X = self.df.drop(self.target_column, axis=1)
            y = self.df[self.target_column]
            self.pls = PLSRegression(n_components=n_components)
            self.pls.fit(X, y)
            self.pls_scores = self.pls.transform(X)

    def plot_pls_results(self, n_components=2):
        self.perform_pls(n_components)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.pls_scores[:, 0], y=self.pls_scores[:, 1], mode='markers',
                                 marker=dict(color=self.df[self.target_column]), text=self.df.index))
        fig.update_layout(title="PLS (2 components)", xaxis_title="PLS 1", yaxis_title="PLS 2")
        fig.show()
    
    def perform_umap(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', force=False):
        """
        UMAP 알고리즘을 사용하여 데이터를 저차원으로 축소합니다.
        이 방법은 데이터의 복잡한 구조를 보존하면서 고차원 데이터를 저차원으로 효과적으로 매핑합니다.

        Args:
            n_components (int): 목표 차원 수입니다.
            n_neighbors (int): 근접 이웃의 수입니다. 이 값은 결과의 구조에 영향을 줍니다.
            min_dist (float): 점들 사이의 최소 거리입니다. 이 값이 클수록 더 넓게 분포합니다.
            metric (str): 거리 계산에 사용할 메트릭입니다.
            force (bool): 기존에 계산된 UMAP 결과를 무시하고 새로 계산할지 여부입니다.

        Returns:
            None: 계산된 UMAP 결과는 클래스의 내부 상태에 저장합니다.
        """
    
        if not hasattr(self, 'umap_results') or force:
            X = self.df.drop(self.target_column, axis=1)
            self.umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
            self.umap_results = self.umap_model.fit_transform(X)

    def plot_umap_results(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        self.perform_umap(n_components, n_neighbors, min_dist, metric)
        fig = px.scatter(x=self.umap_results[:, 0], y=self.umap_results[:, 1], color=self.df[self.target_column], title="UMAP Visualization")
        fig.show()

    def perform_vif(self, force=False, vif_threshold=10.0):
        """
        VIF (Variance Inflation Factor)를 계산하여 다중공선성을 평가하는 함수입니다.
        VIF 값이 높은 변수는 다른 변수와의 강한 상관관계를 가지고 있으며, 이는 회귀 분석 등에서 문제를 일으킬 수 있습니다.
        VIF 값이 임계값을 초과하는 변수는 다르게(Red) 표시하여 다중공선성이 높은 변수를 쉽게 식별할 수 있습니다.

        Args:
            vif_threshold (float): VIF 임계값, 이 값 이상일 경우 다중공선성이 높다고 판단합니다.
            force (bool): True로 설정하면 이미 계산된 VIF 데이터를 무시하고 재계산합니다.

        Returns:
            None: 계산된 VIF 값을 클래스의 내부 상태로 저장합니다.
        """
        if not hasattr(self, 'vif_data') or force:
            X = self.df.drop(self.target_column, axis=1)

            # 상수값 열 제거
            X = X.loc[:, X.apply(pd.Series.nunique) != 1]

            # 완벽한 상관관계가 있는 변수 확인 및 제거
            self.correlated_features = set()
            correlation_matrix = X.corr()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.9:  # 상관계수가 0.9 이상인 경우
                        colname = correlation_matrix.columns[i]
                        self.correlated_features.add(colname)

            X.drop(labels=self.correlated_features, axis=1, inplace=True)

            # VIF 계산
            self.vif_data = pd.DataFrame()
            self.vif_data["feature"] = X.columns
            self.vif_data["VIF"] = [self._calculate_vif_single(X, i) for i in range(X.shape[1])]

            # VIF 값이 높은 특성 식별
            self.high_vif_features = self.vif_data[self.vif_data["VIF"] > vif_threshold]["feature"].tolist()

    def _calculate_vif_single(self, X, index):
        try:
            return variance_inflation_factor(X.values, index)
        except Exception as e:
            return None  # 오류 발생 시 None 반환

    def plot_vif_result(self, vif_threshold=10.0, verbose=False):
        if not hasattr(self, 'vif_data'):
            self.perform_vif(vif_threshold=vif_threshold)

        fig = px.bar(self.vif_data, x='feature', y='VIF', title='Variance Inflation Factor (VIF) for Each Feature')
        fig.add_hline(y=vif_threshold, line_dash="dash", line_color="red")

        # VIF 임계값을 초과하는 변수들의 색상을 변경
        colors = ['red' if feature in self.high_vif_features else 'blue' for feature in self.vif_data['feature']]
        fig.data[0].marker.color = colors

        fig.show()

        if verbose:
            print("Highly Correlated Features (Correlation > 0.9):", self.correlated_features)
            print("Features with High VIF (VIF > 10):", self.high_vif_features)


    def perform_shapiro_wilk_test(self):
        """
        Shapiro-Wilk 테스트를 사용하여 데이터셋의 각 수치형 변수가 정규 분포를 따르는지 평가합니다.
        이 테스트는 작은 표본에 대해서도 정규성을 잘 판단합니다.
        테스트 결과에는 각 변수의 통계값, p-value, 정규성 여부가 포함됩니다.
        필터링 옵션을 통해 정규 또는 비정규 분포를 따르는 변수들만 선택적으로 보여줄 수 있습니다.


        Returns:
            None: Shapiro-Wilk 테스트 결과를 클래스의 내부 상태에 저장합니다.
        """
        self.shapiro_results = pd.DataFrame(columns=['Feature', 'Statistic', 'p-value', 'Normality'])

        for column in self.df.columns:
            if self.df[column].dtype in ['float64', 'int64']:  # 수치형 데이터에 대해서만 테스트
                stat, p = shapiro(self.df[column])
                normality = 'Yes' if p > 0.05 else 'No'
                self.shapiro_results = self.shapiro_results.append({'Feature': column,
                                                                    'Statistic': stat,
                                                                    'p-value': p,
                                                                    'Normality': normality}, ignore_index=True)

    def display_shapiro_results(self, filter_normality=None):
        if not hasattr(self, 'shapiro_results'):
            self.perform_shapiro_wilk_test()

        filtered_results = self.shapiro_results
        if filter_normality in ['normal', 'not-normal']:
            normality_flag = 'Yes' if filter_normality == 'normal' else 'No'
            filtered_results = self.shapiro_results[self.shapiro_results['Normality'] == normality_flag]

        print(filtered_results)
        print("\nInterpretation:")
        for _, row in filtered_results.iterrows():
            print(f"Feature '{row['Feature']}' - Shapiro Statistic: {row['Statistic']:.3f}, p-value: {row['p-value']:.3f}.", end=" ")
            if row['Normality'] == 'Yes':
                print("The distribution is likely normal.")
            else:
                print("The distribution is likely not normal.")
                
    def perform_forward_selection(self, sl_enter=0.05, sl_remove=0.05):
        """
        전진 단계별 선택(Forward Stepwise Selection)을 수행하는 함수입니다.
        이 방법은 변수를 하나씩 추가하며 모델의 성능을 평가합니다.
        p-value가 낮은 변수부터 차례대로 추가하고, 모델의 성능이 더 이상 개선되지 않을 때 멈춥니다.
        각 단계별로 선택된 변수들과 조정된 R² 값을 그래프로 나타냅니다.

        Args:
            sl_enter (float): 변수를 모델에 추가하기 위한 p-value의 최대 임계값입니다.
            sl_remove (float): 모델에서 변수를 제거하기 위한 p-value의 최소 임계값입니다.

        Returns:
            None: 계산된 결과는 클래스의 내부 상태에 저장합니다.
        """
        variables = self.df.columns.drop(self.target_column).tolist()
        y = self.df[self.target_column]
        selected_variables = []
        steps, adjusted_r_squared, sv_per_step = [], [], []

        step = 0
        total_variables = len(variables)
        for var in range(total_variables):
            print(f"Progress: Step {var+1}/{total_variables}")
            remainder = list(set(variables) - set(selected_variables))
            pval = pd.Series(dtype='float64', index=remainder)

            for col in remainder:
                X = self.df[selected_variables + [col]]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                pval[col] = model.pvalues[col]
            
            min_pval = pval.min()
            if min_pval < sl_enter:
                selected_variables.append(pval.idxmin())
                while len(selected_variables) > 0:
                    selected_X = self.df[selected_variables]
                    selected_X = sm.add_constant(selected_X)
                    selected_pval = sm.OLS(y, selected_X).fit().pvalues[1:]
                    max_pval = selected_pval.max()
                    if max_pval >= sl_remove:
                        remove_variable = selected_pval.idxmax()
                        selected_variables.remove(remove_variable)
                    else:
                        break
                adj_r_squared = sm.OLS(y, sm.add_constant(self.df[selected_variables])).fit().rsquared_adj
                adjusted_r_squared.append(adj_r_squared)
                sv_per_step.append(selected_variables.copy())
                steps.append(step)
                step += 1
            else:
                break

        self.forward_selection_results = {'steps': steps, 'adjusted_r_squared': adjusted_r_squared, 'selected_variables': sv_per_step}

    def plot_forward_selection_results(self, sl_enter=0.05, sl_remove=0.05):
        if not hasattr(self, 'forward_selection_results'):
            self.perform_forward_selection(sl_enter, sl_remove)

        # 저장된 결과를 바탕으로 시각화
        fig = go.Figure()

        for step, adj_r_squared in zip(self.forward_selection_results['steps'], self.forward_selection_results['adjusted_r_squared']):
            hover_text = ', '.join(self.forward_selection_results['selected_variables'][step])
            fig.add_trace(go.Scatter(x=[step], y=[adj_r_squared], mode='markers',
                                     hoverinfo='text', text=hover_text))

        fig.update_layout(title='Forward Selection Results',
                          xaxis=dict(title='Step'),
                          yaxis=dict(title='Adjusted R Squared'),
                          showlegend=False)
        fig.show()

        # 최종 선택된 특성들을 출력
        if self.forward_selection_results['selected_variables']:
            final_features = self.forward_selection_results['selected_variables'][-1]
            print("Final selected features:", final_features)

    def perform_influence_analysis(self):
        """
        회귀 모델에서 각 데이터 포인트의 영향력을 분석하는 함수입니다.
        이 함수는 DFFITS, Cook's Distance, DFBETAS와 같은 영향점 지표들을 계산합니다.
        이러한 지표들은 데이터 포인트가 회귀 모델의 추정치에 미치는 영향력을 평가하는 데 사용됩니다.
        각 데이터 포인트가 회귀 모델 추정치에 미치는 영향을 그래프로 나타냅니다.

        Returns:
            None: 계산된 영향점 지표들은 클래스의 내부 상태에 저장합니다.
        """
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # 선형 회귀 모델 피팅
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # 영향점 분석
        influence = model.get_influence()
        self.influence_analysis = pd.DataFrame()

        # DFFITS, Cook's Distance, DFBETAS 계산
        self.influence_analysis['DFFITS'] = influence.dffits[0]
        self.influence_analysis['D'] = influence.cooks_distance[0]
        dfbetas_columns = ['DFBETAS_' + col for col in X.columns]
        self.influence_analysis = pd.concat([self.influence_analysis, pd.DataFrame(influence.dfbetas, columns=dfbetas_columns)], axis=1)

    def display_influence_analysis(self):
        if not hasattr(self, 'influence_analysis'):
            self.perform_influence_analysis()

        n = len(self.df)  # 데이터 포인트의 개수
        p = len(self.df.columns) - 1  # 변수의 개수 (타겟 변수 제외)

        # DFFITS 기준
        dffits_threshold = 2 * np.sqrt(p / n)

        # Cook's Distance 기준
        cooks_d_threshold = 0.5  # 누적 확률값 0.5 or 1이상

        # DFBETAS 기준
        dfbetas_threshold = 2 / np.sqrt(n)

        # DFFITS 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.influence_analysis['DFFITS'], mode='markers', name='DFFITS'))
        fig.add_hline(y=dffits_threshold, line_dash="dash", line_color="red")
        fig.add_hline(y=-dffits_threshold, line_dash="dash", line_color="red")
        fig.update_layout(title='DFFITS (Threshold: ±' + str(dffits_threshold) + ')', xaxis_title='Index', yaxis_title='DFFITS')
        fig.show()

        # Cook's Distance 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.influence_analysis['D'], mode='markers', name="Cook's Distance"))
        fig.add_hline(y=cooks_d_threshold, line_dash="dash", line_color="red")
        fig.update_layout(title="Cook's Distance (Threshold: " + str(cooks_d_threshold) + ")", xaxis_title='Index', yaxis_title="Cook's Distance")
        fig.show()

        # DFBETAS 시각화 (상위 5개 변수)
        dfbetas_cols = [col for col in self.influence_analysis.columns if col.startswith('DFBETAS_')]
        for col in dfbetas_cols[:5]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=self.influence_analysis[col], mode='markers', name=col))
            fig.add_hline(y=dfbetas_threshold, line_dash="dash", line_color="red")
            fig.add_hline(y=-dfbetas_threshold, line_dash="dash", line_color="red")
            fig.update_layout(title=col + ' (Threshold: ±' + str(dfbetas_threshold) + ')', xaxis_title='Index', yaxis_title=col)
            fig.show()

        # 영향점 기준 초과 데이터 포인트 저장
        self.influential_points = {
            'DFFITS': self.influence_analysis[self.influence_analysis['DFFITS'].abs() > dffits_threshold].index.tolist(),
            "Cook's Distance": self.influence_analysis[self.influence_analysis['D'] > cooks_d_threshold].index.tolist(),
            'DFBETAS': {col: self.influence_analysis[self.influence_analysis[col].abs() > dfbetas_threshold].index.tolist() for col in dfbetas_cols}
        }

    def combine_feature_importance(self):
        combined_importance = {}

        # RFE 중요도
        if hasattr(self, 'rfe_features'):
            for feature, is_selected in self.rfe_features.iteritems():
                if is_selected:
                    combined_importance[feature] = combined_importance.get(feature, 0) + 1

        # VIF 분석
        if hasattr(self, 'vif_data'):
            for feature in self.vif_data[self.vif_data['VIF'] < 10]['feature']:
                combined_importance[feature] = combined_importance.get(feature, 0) + 1

        # 상관관계 분석
        corr_matrix = self.df.corr()
        high_corr_threshold = 0.8
        for col in corr_matrix.columns:
            if corr_matrix[col].abs().max() < high_corr_threshold:
                combined_importance[col] = combined_importance.get(col, 0) + 1

        # 가장 중요한 특성 추출
        top_features = sorted(combined_importance, key=combined_importance.get, reverse=True)
        return top_features

    def analyze_clusters(self, n_clusters=3):
        from sklearn.cluster import KMeans
        import seaborn as sns

        # 데이터 클러스터링
        X = self.df.drop(self.target_column, axis=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        self.df['Cluster'] = kmeans.labels_

        # 클러스터별 패턴 분석
        cluster_patterns = self.df.groupby('Cluster').mean()

        # 클러스터별 평균 특성 값 시각화
        sns.clustermap(cluster_patterns, cmap='coolwarm', standard_scale=1)
        return cluster_patterns

    def get_top_bottom_features(self, cluster_patterns, percentile=20):
        # cluster_patterns = self.analyze_clusters()  # 클러스터별 평균 패턴 얻기

        top_features = {}
        bottom_features = {}

        for cluster in cluster_patterns.index:
            cluster_data = cluster_patterns.loc[cluster]

            # 상위 및 하위 퍼센타일 계산
            top_threshold = np.percentile(cluster_data, 100 - percentile)
            bottom_threshold = np.percentile(cluster_data, percentile)

            # 상위 및 하위 피처 추출
            top_features[cluster] = cluster_data[cluster_data >= top_threshold].index.tolist()
            bottom_features[cluster] = cluster_data[cluster_data <= bottom_threshold].index.tolist()

        return top_features, bottom_features

    def analyze_clusters_n(self, n_clusters=3):
        from sklearn.cluster import KMeans
        import seaborn as sns

        # 데이터 클러스터링
        X = self.df.drop(self.target_column, axis=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        self.df['Cluster'] = kmeans.labels_
        cluster_df = pd.DataFrame({'Cluster': kmeans.labels_}, index=self.df.index)
        # 클러스터별 패턴 분석
        cluster_patterns = self.df.groupby('Cluster').mean()
        
        # mean/std-epsilon 스케일링
        cluster_patterns = (cluster_patterns - cluster_patterns.mean()) / (cluster_patterns.std() + 1e-10)

        top_features, bottom_features = self.get_top_bottom_features(cluster_patterns)
        # 클러스터별 평균 특성 값 시각화
        sns.clustermap(cluster_patterns, cmap='coolwarm')
        
        return cluster_patterns

#%%
if __name__== "__main__":
    df = pd.read_pickle('uci_har_dataset.pkl')
    eda = EDA(df, target_column='Activity', drop_columns=None, mini=True)
    # eda.display_basic_info()
    # eda.plot_correlation_matrix()
    # eda.plot_rfe_results()
    # eda.plot_pca_2d_results()
    # eda.plot_pca_3d_results()
    # eda.plot_pls_results()
    # eda.plot_umap_results()
    # eda.plot_vif_result()
    # eda.display_shapiro_results(filter_normality='normal')
    # eda.plot_forward_selection_results()
    # eda.display_influence_analysis()
    top_features = eda.combine_feature_importance()
    eda.analyze_clusters()



# %%
def filter_columns(df, start_with=None, end_with=None, contains=None):
    filtered_cols = df.columns.tolist()

    # 시작 문자열 리스트로 필터링
    if start_with:
        if not isinstance(start_with, list):
            start_with = [start_with]
        filtered_cols = [col for col in filtered_cols if any(col.startswith(start) for start in start_with)]

    # 끝 문자열 리스트로 필터링
    if end_with:
        if not isinstance(end_with, list):
            end_with = [end_with]
        filtered_cols = [col for col in filtered_cols if any(col.endswith(end) for end in end_with)]

    # 특정 문자열 포함 필터링
    if contains:
        if not isinstance(contains, list):
            contains = [contains]
        filtered_cols = [col for col in filtered_cols if any(contain in col for contain in contains)]

    return df[filtered_cols]
#%%
# 함수 검증을 위한 예시 데이터프레임
columns_example = ['X_A01_E0000_LI2CO3', 'X_A02_E0030_TI2LS2', 'X_B01_E0002_ZRD25S',
                   'X_B02_E0000_ZRD25S', 'X_C01_E0040_LI2CO3', 'X_C02_E0001_LI2CO3']
data_example = [[1, 2, 3, 4, 5, 6]]
df_example = pd.DataFrame(data_example, columns=columns_example)

# 다양한 조건으로 함수 검증
filtered_df1 = filter_columns(df_example, start_with=['X_A', 'X_B'], include_numeric=True)
filtered_df2 = filter_columns(df_example, end_with='LI2CO3', include_numeric=False)
filtered_df3 = filter_columns(df_example, start_with='X_C', end_with=['ZRD25S', 'LI2CO3'], include_numeric=True)

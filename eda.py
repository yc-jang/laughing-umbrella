import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import plotly.express as px

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
        corr_matrix = self.df.corr()
        fig = px.imshow(corr_matrix, title="Correlation Matrix")
        fig.show()

    def perform_rfe(self, n_features=5, force=False):
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

    def perform_pca(self, n_components=2, force=False):
        if not hasattr(self, 'pca_df') or force:
            X = self.df.drop(self.target_column, axis=1)
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X)

            # PCA 결과 DataFrame 생성 및 인덱스 맞추기
            self.pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
            self.pca_df.index = self.df.index  # 원본 DataFrame의 인덱스를 pca_df에 복사
            self.pca_df['target'] = self.df[self.target_column]

    def plot_pca_results(self, n_components=2):
        self.perform_pca(n_components)
        fig = px.scatter(self.pca_df, x='PC1', y='PC2', color='target', title="PCA (2 components)")
        fig.show()

    def perform_pls(self, n_components=2, force=False):
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
        if not hasattr(self, 'umap_results') or force:
            X = self.df.drop(self.target_column, axis=1)
            self.umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
            self.umap_results = self.umap_model.fit_transform(X)

    def plot_umap_results(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        self.perform_umap(n_components, n_neighbors, min_dist, metric)
        fig = px.scatter(x=self.umap_results[:, 0], y=self.umap_results[:, 1], color=self.df[self.target_column], title="UMAP Visualization")
        fig.show()
        
if __name__== "__main__":
    df = pd.read_pickle('uci_har_dataset.pkl')
    eda = EDA(df, 'Activity')
    eda.display_basic_info()
    eda.plot_correlation_matrix()
    eda.plot_rfe_results()
    eda.plot_pca_results()
    eda.plot_pls_results()
    eda.plot_umap_results()

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import plotly.express as px

class EDA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = None

    def load_data(self):
        if self.dataset == 'breast_cancer':
            data = load_breast_cancer()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df['target'] = data.target
        else:
            raise ValueError("Unsupported dataset")
    
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

    def perform_rfe(self, n_features=5):
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=5000)
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        self.rfe_features = pd.Series(rfe.support_, index=X.columns)
        return self.rfe_features

    def plot_rfe_results(self):
        fig = px.bar(self.rfe_features, title="Feature Importance (RFE)")
        fig.show()

    def perform_pca(self, n_components=2):
        X = self.df.drop('target', axis=1)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)
        self.pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
        self.pca_df['target'] = self.df['target']

    def plot_pca_results(self):
        fig = px.scatter(self.pca_df, x='PC1', y='PC2', color='target', title="PCA (2 components)")
        fig.show()

# 사용 예시
eda = EDA('breast_cancer')
eda.load_data()
eda.display_basic_info()  # 데이터의 기본 정보를 표시
eda.plot_correlation_matrix()
rfe_features = eda.perform_rfe()
eda.plot_rfe_results()
eda.perform_pca()
eda.plot_pca_results()

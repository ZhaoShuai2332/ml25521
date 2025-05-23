import pandas as pd
import os, sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class CreditFetcher:
    def __init__(self):
        self.credit_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data","datasets", "credit_risk")
        self.X = None
        self.y = None

    def preprocess_features(
        self,
        features: pd.DataFrame,
        test_features: pd.DataFrame = None,
        encoding: str = 'ohe',
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对训练集做：去 ID/标签、编码、列对齐（可选 test_features），然后基于 features 拆分训练/测试集。
        test_features 仅用于列对齐，返回值 x_train, x_test, y_train, y_test 均来自 features。
        
        参数
        ----
        features : pd.DataFrame
            包含 SK_ID_CURR, TARGET 以及特征的完整数据集
        test_features : pd.DataFrame, optional
            包含 SK_ID_CURR（可选 TARGET）的数据集，用于对齐特征列
        encoding : {'ohe', 'le'}
            'ohe'：pd.get_dummies + align
            'le' ：LabelEncoder（仅对 object 列）
        test_size : float
            测试集比例
        random_state : int
            随机种子
        stratify : bool
            是否按 TARGET 做分层抽样
        
        返回
        ----
        x_train, x_test, y_train, y_test : np.ndarray
        """
        # 1. 提取标签，并丢弃 ID/标签 列
        y = features['TARGET'].values
        X = features.drop(columns=['SK_ID_CURR', 'TARGET'])

        # 如果提供了 test_features，则也丢弃其 ID/TARGET
        if test_features is not None:
            X_test_align = test_features.drop(columns=[col for col in ['SK_ID_CURR', 'TARGET'] if col in test_features.columns])
        else:
            X_test_align = None

        # 2. 编码
        if encoding == 'ohe':
            X = pd.get_dummies(X)
            if X_test_align is not None:
                X_test_align = pd.get_dummies(X_test_align)
                # 对齐特征列
                X, X_test_align = X.align(X_test_align, join='inner', axis=1)
        elif encoding == 'le':
            le = LabelEncoder()
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = le.fit_transform(X[col].astype(str))
                    if X_test_align is not None and col in X_test_align.columns:
                        X_test_align[col] = le.transform(X_test_align[col].astype(str))
        else:
            raise ValueError("encoding must be 'ohe' or 'le'")

        # 3. 基于 features 做 train/test 划分
        stratify_arr = y if stratify else None
        x_train, x_test, y_train, y_test = train_test_split(
            X.values, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr
        )

        return x_train, x_test, y_train, y_test


    def load_data(self, encoding='ohe'):
        features = pd.read_csv(os.path.join(self.credit_path, "targeted_data","train_credit.csv"))
        test_features= pd.read_csv(os.path.join(self.credit_path, "targeted_data","test_credit.csv"))   
        features = features.fillna(0)
        test_features = test_features.fillna(0)
        return self.preprocess_features(features, test_features, encoding=encoding)
    

if __name__ == "__main__":
    credit_fetcher = CreditFetcher()
    X_train, X_test, y_train, y_test = credit_fetcher.load_data(encoding='ohe')
        

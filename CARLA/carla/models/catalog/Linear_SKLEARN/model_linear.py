import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Union
import pandas as pd


class LinearModel:
    def __init__(self, dim_input: int, num_of_classes: int):
        """
        Parameters
        ----------
        dim_input: int > 0
            Number of input features
        num_of_classes: int > 0
            Number of output classes
        """
        self.dim_input = dim_input
        self.num_of_classes = num_of_classes
        self.scaler = StandardScaler()
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model to the data.
        :param X: np.ndarray, shape (N, M) where N is the number of samples and M is the number of features
        :param y: np.ndarray, shape (N,) where N is the number of samples
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Predicts the class labels for the given data.
        :param X: np.ndarray or pd.DataFrame, shape (N, M)
        :return: np.ndarray of predictions, shape (N,)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Predicts the probabilities for each class.
        :param X: np.ndarray or pd.DataFrame, shape (N, M)
        :return: np.ndarray of probabilities, shape (N, num_of_classes)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

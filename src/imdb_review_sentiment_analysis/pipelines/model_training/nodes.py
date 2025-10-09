import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def split_data(features_target: tuple) -> tuple:
    X, y = features_target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(classifier: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = classifier.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc_score': roc_auc_score(y_test, y_pred)
    }
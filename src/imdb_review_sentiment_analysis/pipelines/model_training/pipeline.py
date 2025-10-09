from kedro.pipeline import Node, Pipeline
from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=split_data, inputs='tfidf_features', outputs=['X_train', 'X_test', 'y_train', 'y_test'], name='split_data_node'),
        Node(func=train_model, inputs=['X_train', 'y_train'], outputs='logistic_regression_model', name='model_training_node'),
        Node(func=evaluate_model, inputs=['logistic_regression_model', 'X_test', 'y_test'], outputs='logistic_regression_metrics', name='evaluate_model_node'),
    ])

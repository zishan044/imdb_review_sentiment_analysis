from kedro.pipeline import Node, Pipeline
from .nodes import get_features_bag_of_words, get_features_tfidf, get_features_word2vec

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=get_features_bag_of_words, inputs='clean_data', outputs='bag_of_words_features', name='get_features_bag_of_words_node'),
        Node(func=get_features_tfidf, inputs='clean_data', outputs='tfidf_features', name='get_features_tfidf_node'),
        Node(func=get_features_word2vec, inputs='clean_data', outputs='word2vec_features', name='get_features_word2vec_node'),
    ])

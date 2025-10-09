import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

def get_features_bag_of_words(df: pd.DataFrame) -> tuple:
    cnt_vect = CountVectorizer(max_features=5000, min_df=2, max_df=0.95)
    cnt_mat = cnt_vect.fit_transform(df['processed_text'])
    
    X = cnt_mat
    y = df['pos_senti']
    
    return X, y

def get_features_tfidf(df: pd.DataFrame) -> tuple:
    tfidf_vect = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    tfidf_mat = tfidf_vect.fit_transform(df['processed_text'])
    
    X = tfidf_mat
    y = df['pos_senti']
    
    return X, y

def get_features_word2vec(df: pd.DataFrame) -> tuple:
    sentences = [text.split() for text in df['processed_text']]
    
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )
    
    def get_document_vector(words, model):
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    X_w2v = np.array([get_document_vector(words, w2v_model) for words in sentences])
    y = df['pos_senti']
    
    return X_w2v, y
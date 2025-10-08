import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(x: pd.Series) -> pd.Series:
    return (
        x.str.lower()
         .str.replace(r'<.*?>', ' ', regex=True)
         .str.replace(r'http\S+|www\S+', ' ', regex=True)
         .str.replace(r'[^a-z\s.!?]', ' ', regex=True)
         .str.replace(r'\s+', ' ', regex=True)
         .str.strip()
    )

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df_process = df.copy()

    df_process['review'] = clean_text(df_process['review'])

    df_process['sentences'] = df_process['review'].apply(sent_tokenize)
    df_process['words'] = df_process['review'].apply(word_tokenize)

    lemmatizer = WordNetLemmatizer()
    df_process['lemmas'] = df_process['words'].apply(lambda tokens: [lemmatizer.lemmatize(t) for t in tokens])
    
    df_process['pos_senti'] = df_process['sentiment'] == 'positive'
    
    df_process = df_process.drop(columns=['id', 'sentiment'])
    return df_process

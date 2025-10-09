import pandas as pd
import nltk
from multiprocessing import Pool, cpu_count
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', "n't"}

def remove_stopwords(text):
    return ' '.join([w for w in text.split(' ') if w not in stop_words])

def get_wordnet_pos(tag: str) -> str:
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(x: pd.Series) -> pd.Series:
    return (
        x.str.lower()
         .str.replace(r'<.*?>', ' ', regex=True)
         .str.replace(r'http\S+|www\S+', ' ', regex=True)
         .str.replace(r'[^a-z\s.!?]', ' ', regex=True)
         .str.replace(r'\s+', ' ', regex=True)
         .str.strip()
    )

def lemmatize_text(text: str) -> str:
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos_tags]
    return " ".join(lemmas)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_process = df.copy()

    df_process['review'] = clean_text(df_process['review'])
    df_process['lemmatized_text'] = df_process['review'].apply(lemmatize_text)
    
    with Pool(cpu_count()) as p:
        df_process['processed_text'] = p.map(remove_stopwords, df_process['lemmatized_text'])

    df_process['pos_senti'] = (df_process['sentiment'] == 'positive').astype(int)

    return df_process[['processed_text', 'pos_senti']]
import os
import logging
from pathlib import Path
import pandas as pd

logger=logging.getLogger(__name__)

def extract_data_from_files() -> pd.DataFrame:
    path = Path('data/01_raw').resolve()
    records = []
    for folder in path.iterdir():
        if folder.name == 'pos':
            files = folder.glob('*.txt')
            for f in files:
                id, rating = map(int, f.name[:-4].split('_'))
                review = f.read_text()
                records.append({
                    'id': id,
                    'rating': rating,
                    'review': review,
                    'sentiment': 'positive'
                })
        else:
            files = folder.glob('*.txt')
            for f in files:
                id, rating = map(int, f.name[:-4].split('_'))
                review = f.read_text()
                records.append({
                    'id': id,
                    'rating': rating,
                    'review': review,
                    'sentiment': 'negative'
                })
    df = pd.DataFrame(records)
    logger.info(df.info())
    return df
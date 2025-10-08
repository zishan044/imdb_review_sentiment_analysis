from pathlib import Path
import pandas as pd

def extract_data_from_files(base_path: str = 'data/01_raw') -> pd.DataFrame:
    base_path = Path(base_path).resolve()
    records = [
        {
            'id': int(f.stem.split('_')[0]),
            'rating': int(f.stem.split('_')[1]),
            'review': f.read_text(encoding='utf-8'),
            'sentiment': 'positive' if f.parent.name == 'pos' else 'negative'
        }
        for f in base_path.glob('*/*.txt')
    ]
    return pd.DataFrame(records)

"""Download the frozen universe; write a resumable per-symbol audit."""
import sys
import warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
warnings.filterwarnings('ignore',category=FutureWarning)
import pandas as pd
from src.review_revision import download_universe

symbols = pd.read_csv(ROOT/'data/versions/v1.0.0_coursework/sp500_companies.csv').Symbol.tolist()
prices,index,audit = download_universe(symbols,ROOT/'data/versions/yfinance_review_2026_11_01',retries=1)
print(audit.status.value_counts().to_string())
print(audit.query("status != 'ok'").to_string(index=False))
print('Downloaded',prices.Symbol.nunique(),'of',len(symbols),'stocks')

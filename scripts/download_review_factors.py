import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.review_factors import prepare_factors

if __name__=='__main__':
    data=prepare_factors(ROOT/'data/versions/french_factors_review_2026_11_01',download=True)
    print('FF5+momentum:',len(data),'dates',data.Date.min(),data.Date.max())

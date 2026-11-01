import numpy as np
import pandas as pd
import pytest

from src import review_factors as rf


def test_daily_french_parser_converts_percent_and_excludes_footer():
    raw='Header\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20240102,1.0,2,3,4,5,0.01\n20240103,-1,0,0,0,0,0.02\nAnnual Factors: January-December\n2024,5,6,7,8,9,1\n'
    out=rf.parse_french_daily(raw)
    assert len(out)==2
    assert out['Mkt-RF'].iloc[0]==pytest.approx(.01)
    assert out.RF.iloc[1]==pytest.approx(.0002)
    assert out.Date.iloc[0]==pd.Timestamp('2024-01-02')


def test_factor_coverage_never_silently_drops_a_trading_day():
    factors=pd.DataFrame({'Date':pd.to_datetime(['2024-01-02']), 'RF':[.0001],
                          'Mkt-RF':[.01],'SMB':[0],'HML':[0],'RMW':[0],'CMA':[0],'Mom':[0]})
    with pytest.raises(ValueError,match='cover'):
        rf.require_coverage(factors,pd.to_datetime(['2024-01-02','2024-01-03']))

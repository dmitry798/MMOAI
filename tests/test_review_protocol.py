import numpy as np
import pandas as pd
import pytest
from src import review_protocol as rp


def test_saved_prediction_csv_roundtrip_needs_no_parquet_engine(tmp_path):
    frame=pd.DataFrame({'Date':pd.to_datetime(['2024-01-02']), 'Symbol':['A'], 'y_prob':[.123]})
    path=rp.save_predictions(frame,tmp_path/'predictions')
    assert str(path).endswith('.csv.gz')
    pd.testing.assert_frame_equal(pd.read_csv(path,parse_dates=['Date']),frame)


def test_run_integrity_rejects_trade_exiting_outside_evaluation():
    dates=pd.bdate_range('2020-01-01',periods=5)
    result=dict(params={'capital':100}, equity=pd.DataFrame({'Capital':[100]*5},index=dates),
                trades=pd.DataFrame({'SignalDate':[dates[0]],'EntryDate':[dates[1]],
                                     'ExitDate':[dates[-1]+pd.Timedelta(days=3)],'Net PnL ($)':[0]}))
    with pytest.raises(AssertionError,match='exit'):
        rp.validate_backtest(result,dates[0],dates[-1])

"""Regression tests for scientific errors identified during review."""
import numpy as np
import pandas as pd
import pytest

from src import review_revision as r


def test_sha256_works_without_python311_file_digest(tmp_path, monkeypatch):
    monkeypatch.delattr(r.hashlib, 'file_digest', raising=False)
    path = tmp_path / 'quotes.csv'
    path.write_bytes(b'abc')
    assert r.sha256(path) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


def panel(n=40):
    dates = pd.bdate_range('2020-01-01', periods=n)
    index = pd.DataFrame({'Date': dates, 'SP500': 100.0})
    prices = pd.DataFrame({'Date': dates, 'Symbol': 'A', 'Close': 100.0})
    return prices, index


def test_unobserved_label_tail_is_unknown_even_after_early_hit():
    prices, index = panel(15)
    prices.loc[12:, 'Close'] = 110
    out = r.build_base_panel(prices, index, horizon=10)
    assert out.Target.notna().sum() == 5
    assert out.Target.iloc[-10:].isna().all()
    assert out.LabelEnd.iloc[0] == index.Date.iloc[10]


def test_missing_session_does_not_compress_stock_horizon():
    prices, index = panel(30)
    prices = prices.drop(index=8)
    out = r.build_base_panel(prices, index, horizon=10)
    assert pd.isna(out.Target.iloc[0])
    assert pd.isna(out.loc[out.Date == index.Date.iloc[9], 'X_1']).all()


def test_cv_never_splits_same_date_and_purges_future_labels():
    prices, index = panel(90)
    df = r.build_base_panel(prices, index).dropna(subset=['Target'])
    df = pd.concat([df.assign(Symbol='A'), df.assign(Symbol='B')]).sort_values('Date').reset_index(drop=True)
    for tr, te in r.date_cv_splits(df, n_splits=3, gap=10):
        assert not set(df.Date.iloc[tr]) & set(df.Date.iloc[te])
        assert df.LabelEnd.iloc[tr].max() < df.Date.iloc[te].min()


def test_monthly_schedules_isolate_window_shape():
    prices, index = panel(100)
    df = r.build_base_panel(prices, index).dropna(subset=['Target'])
    start, end = index.Date.iloc[50], index.Date.iloc[80]
    e = r.walk_plan(df, start, end, window=None, gap=10, step=21)
    s = r.walk_plan(df, start, end, window=20, gap=10, step=21)
    assert len(e) == len(s) == 2
    for a,b in zip(e,s):
        np.testing.assert_array_equal(a['test'], b['test'])
        assert a['train'][-1] == b['train'][-1]
        assert df.LabelEnd.iloc[a['train']].max() < df.Date.iloc[a['test']].min()
        assert df.Date.iloc[b['train']].nunique() == 20
    assert len(e[1]['train']) > len(e[0]['train'])


def test_unknown_future_label_does_not_remove_oos_candidate():
    from sklearn.tree import DecisionTreeClassifier
    prices,index=panel(100)
    prices['Close'] = 100+np.sin(np.arange(100))*4
    data=r.build_base_panel(prices,index)
    data['feature'] = np.arange(100,dtype=float)
    data.loc[70,'Target']=np.nan
    predictions,_=r.walk_predict(data,['feature'],{'tree':DecisionTreeClassifier(max_depth=2)},
                                index.Date.iloc[65],index.Date.iloc[75],window=30)
    assert len(predictions)==11
    assert pd.isna(predictions.loc[predictions.Date==index.Date.iloc[70],'y_true']).all()


def test_bh_adjustment_hand_calculated():
    np.testing.assert_allclose(r.bh_adjust([.01,.04,.03,.2]), [.04,.0533333333,.0533333333,.2])


def test_bootstrap_resamples_consecutive_blocks():
    ix = r.circular_blocks(10, 4, np.random.default_rng(42))
    assert len(ix) == 10
    for block in (ix[:4], ix[4:8], ix[8:]):
        assert ((np.diff(block) % 10) == 1).all()


def test_backtest_records_first_loss_daily_drawdown_and_no_exit_beyond_end():
    prices, index = panel(10)
    prices.loc[2:, 'Close'] = 90
    preds = prices[['Date','Symbol']].assign(y_prob=.9,y_true=0)
    res = r.run_backtest(preds, prices, index, horizon=2,
                         start=index.Date.iloc[0], end=index.Date.iloc[6],
                         top_k=1, commission=0, capital=100)
    assert res['metrics']['final_capital'] == pytest.approx(90)
    assert res['metrics']['max_drawdown'] == pytest.approx(-.1)
    assert res['equity'].Capital.iloc[0] == 100
    assert res['trades'].ExitDate.max() <= index.Date.iloc[6]
    assert res['trades'].EntryDate.min() == index.Date.iloc[1]
    assert res['period_returns'].iloc[0] == pytest.approx(-.1)


def test_constant_price_roundtrip_charges_entry_and_exit_notional():
    prices, index = panel(5)
    preds = prices[['Date','Symbol']].assign(y_prob=.9,y_true=0)
    res = r.run_backtest(preds, prices, index, horizon=2,
                         start=index.Date.iloc[0], end=index.Date.iloc[3],
                         top_k=1, commission=.01, capital=100)
    assert res['metrics']['final_capital'] == pytest.approx(100*.99/1.01)
    assert res['metrics']['avg_turnover'] == pytest.approx(2/1.01)


def test_selection_does_not_drop_winner_with_missing_future_quote():
    prices, index = panel(6)
    b = prices.assign(Symbol='B')
    prices.loc[3, 'Close'] = np.nan
    both = pd.concat([prices,b])
    preds = both[['Date','Symbol']].assign(y_prob=lambda d: np.where(d.Symbol=='A',.9,.1),y_true=0)
    with pytest.raises(ValueError, match='held|Held'):
        r.run_backtest(preds, both, index, horizon=2,
                       start=index.Date.iloc[0], end=index.Date.iloc[4], top_k=1)


def test_missing_entry_quote_keeps_signal_slot_in_cash_without_promoting_next_stock():
    prices,index=panel(5)
    other=prices.assign(Symbol='B')
    prices.loc[1,'Close']=np.nan
    other.loc[2:,'Close']=200
    both=pd.concat([prices,other],ignore_index=True)
    preds=both[['Date','Symbol']].assign(y_prob=lambda d:np.where(d.Symbol=='A',.9,.1),y_true=0)
    result=r.run_backtest(preds,both,index,horizon=2,start=index.Date.iloc[0],
                          end=index.Date.iloc[3],top_k=1,commission=0,capital=100)
    assert result['metrics']['final_capital']==pytest.approx(100)
    assert result['metrics']['n_trades']==0


def test_all_273_features_retain_original_formulas():
    prices, index = panel(300)
    prices['Close'] = np.arange(300)+100.
    df = r.add_statistical_features(r.build_base_panel(prices,index))
    assert len(r.ALL_FEATURES) == 273
    assert df.X_5.iloc[270] == pytest.approx((370/365-1)*100)
    assert df.X_5_std_2.iloc[270] == pytest.approx(np.std([(369/364-1)*100,(370/365-1)*100],ddof=1))


def test_overlapping_cohorts_have_same_daily_calendar_for_both_horizons():
    prices,index = panel(40)
    preds=prices[['Date','Symbol']].assign(y_prob=.9,y_true=0)
    a=r.overlapping_backtest(preds,prices,index,horizon=10,start=index.Date.iloc[0],end=index.Date.iloc[-1],top_k=1,commission=0)
    b=r.overlapping_backtest(preds,prices,index,horizon=21,start=index.Date.iloc[0],end=index.Date.iloc[-1],top_k=1,commission=0)
    pd.testing.assert_index_equal(a['equity'].index,b['equity'].index)
    assert a['metrics']['final_capital'] == pytest.approx(100000)
    assert b['metrics']['final_capital'] == pytest.approx(100000)

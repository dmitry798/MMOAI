import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import review_inference as ri


def test_basic_interval_and_equal_tail_p_agree_for_skewed_errors():
    errors = np.random.default_rng(4).exponential(size=2000) - 1
    for observed in np.linspace(-3, 4, 101):
        out = ri.bootstrap_inference(observed, observed + errors)
        assert (out['p'] < .05) == (out['ci_low'] > 0 or out['ci_high'] < 0)
    # An integer critical rank also uses strict p < alpha.
    for observed in np.linspace(-2, 3, 30):
        out = ri.bootstrap_inference(observed, observed + errors[:1999])
        assert (out['p'] < .05) == (out['ci_low'] > 0 or out['ci_high'] < 0)


def test_degenerate_and_invalid_draws_are_explicit():
    same = ri.bootstrap_inference(0, np.zeros(2000))
    assert same['p'] == 1 and same['ci_low'] == same['ci_high'] == 0
    different = ri.bootstrap_inference(1, np.ones(2000))
    assert different['p'] == pytest.approx(2/2001)
    invalid = ri.bootstrap_inference(1, np.r_[np.ones(1900), np.full(100, np.nan)])
    assert invalid['status'] == 'not_estimable' and np.isnan(invalid['p'])
    assert invalid['n_boot'] == 2000 and invalid['n_valid'] == 1900


def test_holm_preserves_nan_and_original_order():
    np.testing.assert_allclose(ri.holm([.04, np.nan, .001, .02]), [.04, np.nan, .003, .04], equal_nan=True)


def test_bh_preserves_nan_and_original_order():
    np.testing.assert_allclose(ri.bh([.01, .04, np.nan, .03]), [.03, .04, np.nan, .04], equal_nan=True)


def test_blocks_require_at_least_two_possible_date_compositions():
    with pytest.raises(ValueError, match='shorter'):
        ri.circular_blocks(20, 21, np.random.default_rng(0))
    with pytest.raises(ValueError, match='shorter'):
        ri.circular_blocks(20, 20, np.random.default_rng(0))


def auc_panel():
    rng = np.random.default_rng(20)
    dates = pd.date_range('2020-01-01', periods=60)
    y = np.tile([0, 1, 0, 1], len(dates))
    score = np.round(rng.normal(size=len(y)) + y * .8, 1)
    return pd.DataFrame({'Date': np.repeat(dates, 4), 'y_true': y, 'a': score, 'b': score})


def test_auc_shared_date_blocks_ties_and_reproducibility():
    frame = auc_panel()
    a = ri.block_bootstrap_aucs(frame, ['a', 'b'], n_boot=200, block=7)
    b = ri.block_bootstrap_aucs(frame, ['a', 'b'], n_boot=200, block=7)
    assert a['observed']['a'] == pytest.approx(roc_auc_score(frame.y_true, frame.a))
    np.testing.assert_array_equal(a['draws'], b['draws'])
    diff = ri.bootstrap_diff(a, 'a', 'b')
    assert diff['diff'] == diff['ci_low'] == diff['ci_high'] == 0
    assert diff['p'] == 1
    iid = ri.block_bootstrap_aucs(frame, ['a', 'b'], n_boot=200, block=1)
    assert not np.array_equal(a['draws'], iid['draws'])


def test_auc_date_multiplicities_match_full_row_resampling():
    frame = auc_panel()
    # Unequal cross-section sizes distinguish row- from equal-date-weight AUC.
    frame = frame.drop(frame.index[(frame.index % 13) == 0]).reset_index(drop=True)
    out = ri.block_bootstrap_aucs(frame, ['a'], n_boot=200, block=9, seed=19)
    dates = sorted(frame.Date.unique())
    ix = ri.circular_blocks(len(dates), 9, np.random.default_rng(19))
    sampled = pd.concat([frame.loc[frame.Date == dates[i]] for i in ix])
    assert out['draws'][0, 0] == pytest.approx(roc_auc_score(sampled.y_true, sampled.a))


def test_auc_missing_predictions_do_not_silently_change_population():
    frame = auc_panel()
    frame.loc[0, 'a'] = np.nan
    with pytest.raises(ValueError, match='all labeled rows'):
        ri.block_bootstrap_aucs(frame, ['a', 'b'], n_boot=10)


def make_economy(n=300):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range('2020-01-01', periods=n+1)
    market = rng.normal(0, .01, n)
    rf = np.full(n, .0001)
    returns = rf + .0007 + .5 * market
    equity = pd.DataFrame({'Capital': 100*np.r_[1, np.cumprod(1+returns)]}, index=dates)
    trades = pd.DataFrame({'ExitDate': dates[1:], 'Net PnL ($)': np.where(np.arange(n)%2, -1., 2.)})
    result = {'equity': equity, 'trades': trades}
    prices = pd.DataFrame({'Date': dates, 'SP500': 100*np.r_[1, np.cumprod(1+market+rf)]})
    factors = pd.DataFrame({'Date': dates[1:], 'RF': rf, 'Mkt-RF': market})
    return result, prices, factors


def test_identical_economies_have_exact_zero_contrasts():
    result, prices, factors = make_economy()
    out = ri.economic_bootstrap(result, result, prices, factors=factors, n_boot=200)
    assert len(out) == 6
    np.testing.assert_allclose(out.estimate, 0)
    np.testing.assert_allclose(out.ci_low, 0)
    np.testing.assert_allclose(out.ci_high, 0)
    np.testing.assert_allclose(out.p, 1)


def test_economic_tables_can_be_concatenated_for_reporting():
    result, prices, factors = make_economy()
    paired = ri.economic_bootstrap(result, result, prices, factors=factors, n_boot=100)
    single = ri.individual_bootstrap(result, prices, factors=factors, n_boot=100)
    report = pd.concat([paired, single], ignore_index=True)
    assert len(report) == 12


def test_capm_uses_actual_rf_and_refits_intercept():
    result, prices, factors = make_economy()
    out = ri.individual_bootstrap(result, prices, factors=factors, n_boot=200).set_index('metric')
    assert out.loc['jensen_alpha', 'estimate'] == pytest.approx(.0007*252, abs=1e-12)
    assert out.loc['jensen_alpha', 'ci_low'] == pytest.approx(.0007*252, abs=1e-12)
    assert out.loc['jensen_alpha', 'p'] < .05
    assert out.loc['profit_factor', 'estimate'] == 2
    assert out.loc['profit_factor', 'null'] == 1


def test_drawdown_includes_starting_wealth_and_missing_factors_fail():
    result, prices, factors = make_economy(20)
    result['equity']['Capital'] = 100*np.r_[1, np.cumprod(np.full(20, .99))]
    out = ri.individual_bootstrap(result, prices, n_boot=200, block=5).set_index('metric')
    assert out.loc['max_drawdown', 'estimate'] == pytest.approx(.99**20-1)
    with pytest.raises(ValueError, match='factor'):
        ri.individual_bootstrap(result, prices, factors=factors.iloc[1:], n_boot=200)


def test_joint_close_date_pnl_retains_winning_and_losing_trades():
    result, prices, factors = make_economy(40)
    dates = result['equity'].index[1:]
    result['trades'] = pd.DataFrame({'ExitDate': np.repeat(dates, 2),
                                     'Net PnL ($)': np.tile([3., -1.], len(dates))})
    out = ri.individual_bootstrap(result, prices, factors=factors).set_index('metric')
    pf = out.loc['profit_factor']
    assert pf['n_boot'] == 2000
    assert pf['estimate'] == pf['ci_low'] == pf['ci_high'] == 3
    assert pf['p'] == pytest.approx(2/2001)
    assert 'conditional_observed_notional' in pf['estimand']


def test_ff_six_factor_intercept_is_refitted_with_decimal_daily_rf():
    result, prices, factors = make_economy(300)
    rng = np.random.default_rng(40)
    for column in ['SMB', 'HML', 'RMW', 'CMA', 'Mom']:
        factors[column] = rng.normal(0, .01, len(factors))
    coefficients = np.array([.3, -.1, .2, .1, .05, -.2])
    returns = factors.RF.to_numpy()+.0004+factors[list(ri.FF_COLUMNS)].to_numpy() @ coefficients
    result['equity']['Capital'] = 100*np.r_[1, np.cumprod(1+returns)]
    out = ri.individual_bootstrap(result, prices, factors=factors, n_boot=200).set_index('metric')
    ff = out.loc['ff_alpha']
    assert ff['estimate'] == pytest.approx(.0004*252, abs=1e-12)
    assert ff['ci_low'] == pytest.approx(.0004*252, abs=1e-12)
    assert ff['ci_high'] == pytest.approx(.0004*252, abs=1e-12)
    assert ff['p'] < .05


def test_feature_blocks_pooled_class_means_and_zero_contrast():
    dates = pd.bdate_range('2020-01-01', periods=40)
    rows = []
    for i, date in enumerate(dates):
        rows.extend([{'Date': date, 'Target': 0, 'signal': float(i), 'same': 3.}])
        rows.extend([{'Date': date, 'Target': 1, 'signal': float(i+2), 'same': 3.}]*((i % 3)+1))
    frame = pd.DataFrame(rows)
    out = ri.feature_block_tests(frame, ['signal', 'same'], block=5)
    expected = frame.groupby('Target').signal.mean().diff().iloc[-1]
    assert out.loc['signal', 'mean_diff'] == pytest.approx(expected)
    assert out.loc['same', 'mean_diff'] == out.loc['same', 'ci_low'] == out.loc['same', 'ci_high'] == 0
    assert out.loc['same', 'p'] == out.loc['same', 'q_BH'] == 1
    assert out.loc['same', 'n_boot'] == 2000
    assert out.index.name == 'feature'
    np.testing.assert_allclose(out.q_BY, np.minimum(1., out.q_BH*1.5))


def test_feature_numeric_strings_are_rejected_before_string_aggregation():
    dates = pd.bdate_range('2020-01-01', periods=40)
    frame = pd.DataFrame({'Date': np.repeat(dates, 2), 'Target': np.tile([0, 1], len(dates)),
                          'feature': np.tile(['1', '2'], len(dates))})
    with pytest.raises(ValueError, match='numeric'):
        ri.feature_block_tests(frame, ['feature'], n_boot=100)


def test_feature_blocks_retain_unknown_label_dates_and_reproduce():
    dates = pd.bdate_range('2020-01-01', periods=80)
    frame = pd.DataFrame({'Date': np.repeat(dates, 2), 'Target': np.tile([0., 1.], len(dates)),
                          'signal': np.repeat(np.arange(len(dates)), 2) * np.tile([0., 1.], len(dates))})
    frame.loc[frame.Date.isin(dates[::3]), 'Target'] = np.nan
    a = ri.feature_block_tests(frame, ['signal'], n_boot=300, block=8, seed=4)
    b = ri.feature_block_tests(frame, ['signal'], n_boot=300, block=8, seed=4)
    iid = ri.feature_block_tests(frame, ['signal'], n_boot=300, block=1, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc['signal', 'n_days'] == 80
    assert a.loc['signal', 'ci_high']-a.loc['signal', 'ci_low'] > iid.loc['signal', 'ci_high']-iid.loc['signal', 'ci_low']

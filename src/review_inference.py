"""Matched basic bootstrap intervals and tests using synchronized date blocks.

Inference is conditional on the fitted forecasts and observed trading path. It
does not incorporate model selection, refitting forecasts, or a regenerated
portfolio/notional path. Profit factor resamples joint daily positive/negative
realized trade-dollar PnL, assigning each complete cross-section to its close
date. Total-return and drawdown inference is descriptive for this fixed horizon;
annual mean log growth is the more nearly stationary growth estimand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

METHOD = 'circular_moving_block_basic_equal_tail_plus_one'
METRICS = ('annual_log_growth', 'jensen_alpha', 'sharpe', 'max_drawdown',
           'profit_factor', 'total_return')
FF_COLUMNS = ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom')


def bootstrap_inference(observed, draws, null=0., alpha=.05):
    """Invert an equal-tail test of centered bootstrap errors.

    The critical order statistic is ceil(alpha*(B+1)/2)-1, ensuring that
    exclusion of the null agrees with the *strict* test p < alpha, including
    finite B and ties. No interpolation of quantiles is used. Nonfinite draws
    are never silently discarded: any such draw makes inference unavailable.
    n_valid and valid_fraction expose degeneracy (99% is a minimum requirement,
    not permission to condition away failed replicates).
    """
    draws = np.asarray(draws, dtype=float).ravel()
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    b = len(draws)
    valid = int(np.isfinite(draws).sum())
    out = dict(ci_low=np.nan, ci_high=np.nan, p=np.nan, n_valid=valid,
               n_boot=b, alpha=alpha, method=METHOD, status='not_estimable',
               valid_fraction=valid/b if b else 0.)
    if not b or not np.isfinite(observed) or not np.isfinite(null) or valid != b:
        return out
    errors = np.sort(draws - observed)
    # nextafter guards floating-point arithmetic at integer critical ranks.
    rank = int(np.ceil(np.nextafter(alpha*(b+1)/2, -np.inf))) - 1
    if rank < 1:
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = observed-errors[-rank], observed-errors[rank-1]
    distance = observed-null
    p = min(1., 2*(1+min(np.count_nonzero(errors <= distance),
                         np.count_nonzero(errors >= distance)))/(b+1))
    out.update(ci_low=float(lo), ci_high=float(hi), p=float(p), status='ok')
    return out


def holm(pvalues):
    """Holm familywise adjustment over finite p-values, preserving NaNs."""
    p = np.asarray(pvalues, dtype=float)
    result = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    finite = p[mask]
    if ((finite < 0) | (finite > 1)).any():
        raise ValueError('p-values must be in [0, 1]')
    order = np.argsort(finite, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(finite[order]*(len(finite)-np.arange(len(finite)))))
    restored = np.empty_like(finite)
    restored[order] = adjusted
    result[mask] = restored
    return result


def bh(pvalues):
    """Benjamini-Hochberg adjustment over finite tests, retaining NaNs.

    BH controls FDR under independence or appropriate positive dependence;
    shared bootstrap draws do not establish that assumption across features.
    """
    p = np.asarray(pvalues, dtype=float)
    result = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    finite = p[mask]
    if ((finite < 0) | (finite > 1)).any():
        raise ValueError('p-values must be in [0, 1]')
    order = np.argsort(finite, kind='stable')
    adjusted = finite[order]*len(finite)/np.arange(1, len(finite)+1)
    adjusted = np.minimum(1., np.minimum.accumulate(adjusted[::-1])[::-1])
    restored = np.empty_like(finite)
    restored[order] = adjusted
    result[mask] = restored
    return result


def feature_block_tests(frame, feature_names, n_boot=2000, block=21, seed=42, alpha=.05):
    """Pooled class-1 minus class-0 mean contrasts with shared date blocks.

    Aggregate feature sums and class counts by date before resampling, retaining
    the calendar of unknown-label dates as zero-count dates. This preserves
    cross-sectional association and temporal blocks; it does not assume that
    individual stock rows are independent. All requested features must be finite
    on labeled rows. Intervals/tests concern the fixed training population and
    observed labels, without refitting or correcting feature-selection bias.
    q_BH is the requested BH correction; its cross-feature dependence assumptions
    should not be inferred from date-block resampling alone. q_BY additionally
    supplies the conservative arbitrary-dependence FDR adjustment.
    """
    names = list(feature_names)
    if not names or len(names) != len(set(names)) or n_boot < 1:
        raise ValueError('Unique feature names and positive n_boot required')
    target = 'Target' if 'Target' in frame else 'y_true'
    row_dates = pd.to_datetime(frame.Date)
    if frame.empty or row_dates.isna().any():
        raise ValueError('A nonempty, complete date calendar is required')
    dates = pd.DatetimeIndex(sorted(row_dates.unique()))
    labels = frame[target]
    known = labels.notna()
    if not labels[known].isin([0, 1]).all() or labels[known].nunique() != 2:
        raise ValueError('Binary labels with both classes required')
    for name in names:
        if not pd.api.types.is_numeric_dtype(frame[name].dtype):
            raise ValueError('Features must have numeric dtypes before aggregation')
        if not np.isfinite(frame.loc[known, name].to_numpy(dtype=float)).all():
            raise ValueError('All features must be finite on every labeled row')
    sums, counts = [], []
    for label in (0, 1):
        selected = labels.eq(label)
        # Group each class only once. Calendar reindex keeps dates with no
        # labeled rows; counts, rather than day means, preserve pooled weights.
        grouped = frame.loc[selected, names].groupby(row_dates[selected], sort=False)
        sums.append(grouped.sum().reindex(dates, fill_value=0).to_numpy(dtype=float))
        counts.append(grouped.size().reindex(dates, fill_value=0).to_numpy(dtype=float))
    observed = sums[1].sum(axis=0)/counts[1].sum()-sums[0].sum(axis=0)/counts[0].sum()
    rng = np.random.default_rng(seed)
    weights = np.empty((n_boot, len(dates)), dtype=float)
    for i in range(n_boot):
        weights[i] = np.bincount(circular_blocks(len(dates), block, rng), minlength=len(dates))
    with threadpool_limits(limits=1), np.errstate(invalid='ignore', divide='ignore'):
        draws = ((weights @ sums[1])/(weights @ counts[1])[:, None]
                 -(weights @ sums[0])/(weights @ counts[0])[:, None])
    rows = []
    for j, name in enumerate(names):
        rows.append(dict(feature=name, mean_diff=float(observed[j]), null=0.,
                         n_days=len(dates), block=block, seed=seed,
                         n_class0=int(counts[0].sum()), n_class1=int(counts[1].sum()),
                         **bootstrap_inference(observed[j], draws[:, j], alpha=alpha)))
    out = pd.DataFrame(rows).set_index('feature')
    out['q_BH'] = bh(out.p.to_numpy())
    n_finite = int(np.isfinite(out.p).sum())
    harmonic = np.sum(1./np.arange(1, n_finite+1))
    out['q_BY'] = np.minimum(1., out.q_BH*harmonic)
    out.attrs.update(contrast='class1_minus_class0_pooled_mean',
                     resampling='shared_circular_date_blocks',
                     multiplicity='BH requires independence or suitable positive dependence; BY permits arbitrary dependence')
    return out


def circular_blocks(n, block, rng):
    if not isinstance(n, (int, np.integer)) or not isinstance(block, (int, np.integer)) or n < 1 or block < 1:
        raise ValueError('Positive integer n and block required')
    if block >= n:
        raise ValueError('Block must be shorter than the evaluation calendar; full-calendar rotations cannot estimate sampling uncertainty')
    starts = rng.integers(0, n, size=int(np.ceil(n/block)))
    return ((starts[:, None]+np.arange(block)) % n).ravel()[:n]


def _auc_evaluator(y, scores, date_codes):
    """Sort once; weighted Mann-Whitney AUC includes half credit for ties."""
    order = np.argsort(scores, kind='stable')
    sorted_score = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    positive = y[order] == 1
    codes = date_codes[order]

    def evaluate(date_weights):
        weight = date_weights[codes]
        pos = np.add.reduceat(weight*positive, starts)
        neg = np.add.reduceat(weight*(~positive), starts)
        denominator = pos.sum()*neg.sum()
        if denominator <= 0:
            return np.nan
        return float(pos @ (np.cumsum(neg)-.5*neg) / denominator)
    return evaluate


def block_bootstrap_aucs(df, model_names, n_boot=2000, block=21, alpha=.05, seed=42):
    """Pooled row AUC, resampling whole date cross-sections jointly by model.

    Models must have predictions on all labeled rows; missing predictions do
    not redefine the comparison population. Unlabeled rows are excluded before
    defining the evaluation calendar. Final analyses should use n_boot >= 2000.
    """
    names = list(model_names)
    if n_boot < 1 or not names or len(set(names)) != len(names):
        raise ValueError('Positive n_boot and unique model names required')
    frame = df.loc[df.y_true.notna(), ['Date', 'y_true']+names].copy()
    if frame.empty or frame.Date.isna().any():
        raise ValueError('No complete labeled date population')
    y = frame.y_true.to_numpy(dtype=float)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Binary labels with both classes required')
    if not np.isfinite(frame[names].to_numpy(dtype=float)).all():
        raise ValueError('All models require finite predictions on all labeled rows')
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame.Date).unique()))
    codes = dates.get_indexer(pd.to_datetime(frame.Date))
    evaluators = [_auc_evaluator(y, frame[name].to_numpy(dtype=float), codes) for name in names]
    observed = {name: evaluate(np.ones(len(dates))) for name, evaluate in zip(names, evaluators)}
    draws = np.empty((n_boot, len(names)))
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        weights = np.bincount(circular_blocks(len(dates), block, rng), minlength=len(dates))
        draws[i] = [evaluate(weights) for evaluate in evaluators]
    inference = {name: bootstrap_inference(observed[name], draws[:, j], null=.5, alpha=alpha)
                 for j, name in enumerate(names)}
    return dict(observed=observed, draws=draws, model_names=names,
                ci={name: (x['ci_low'], x['ci_high']) for name, x in inference.items()},
                p_vs_half={name: x['p'] for name, x in inference.items()},
                inference=inference, n_boot=n_boot, alpha=alpha, block=block,
                n_days=len(dates), n_rows=len(frame), dates=dates, seed=seed,
                method=METHOD, unit='date_cross_section', resampling='shared_circular_blocks')


def bootstrap_diff(boot, a, b):
    names = boot['model_names']
    observed = boot['observed'][a]-boot['observed'][b]
    draws = boot['draws'][:, names.index(a)]-boot['draws'][:, names.index(b)]
    return dict(diff=observed, estimate=observed, null=0., block=boot['block'],
                n_days=boot['n_days'], **bootstrap_inference(observed, draws, alpha=boot['alpha']))


def _date_frame(frame, label):
    out = frame.copy()
    if 'Date' in out:
        out = out.set_index('Date')
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    if out.index.has_duplicates or out.index.isna().any():
        raise ValueError(f'Duplicate or invalid {label} dates')
    return out.sort_index()


def _economic_inputs(results, index_prices, factors):
    equities = [_date_frame(result['equity'], 'equity') for result in results]
    calendar = equities[0].index
    if len(calendar) < 4 or any(not e.index.equals(calendar) for e in equities):
        raise ValueError('Daily evaluation calendars differ or have fewer than four closes')
    wealth = [e.Capital.to_numpy(dtype=float) for e in equities]
    if any(not np.isfinite(w).all() or (w <= 0).any() for w in wealth):
        raise ValueError('Finite positive daily wealth required')
    dates = calendar[1:]
    market_prices = _date_frame(index_prices, 'market').SP500.reindex(calendar).to_numpy(dtype=float)
    if not np.isfinite(market_prices).all() or (market_prices <= 0).any():
        raise ValueError('Market prices must cover the complete evaluation calendar')
    market = market_prices[1:]/market_prices[:-1]-1
    rf = np.zeros(len(dates))
    ff = None
    source = 'zero_rate_price_return_CAPM_approximation'
    if factors is not None:
        factor_frame = _date_frame(factors, 'factor').reindex(dates)
        rf_name = 'RF' if 'RF' in factor_frame else 'rf'
        if rf_name not in factor_frame or not np.isfinite(factor_frame[rf_name].to_numpy(dtype=float)).all():
            raise ValueError('External factor RF must cover every daily return date')
        rf = factor_frame[rf_name].to_numpy(dtype=float)
        source = 'actual_index_excess_return_CAPM_external_daily_RF'
        if all(c in factor_frame for c in FF_COLUMNS):
            ff = factor_frame[list(FF_COLUMNS)].to_numpy(dtype=float)
            if not np.isfinite(ff).all():
                raise ValueError('External factors must cover every daily return date')
    strategies = []
    for result, w in zip(results, wealth):
        returns = w[1:]/w[:-1]-1
        positive, negative = np.zeros(len(dates)), np.zeros(len(dates))
        trades = result.get('trades', pd.DataFrame())
        if len(trades):
            if not {'ExitDate', 'Net PnL ($)'}.issubset(trades):
                raise ValueError('Realized trades require ExitDate and Net PnL ($)')
            codes = dates.get_indexer(pd.to_datetime(trades.ExitDate))
            pnl = trades['Net PnL ($)'].to_numpy(dtype=float)
            if (codes < 0).any() or not np.isfinite(pnl).all():
                raise ValueError('Every realized trade must close on an evaluation return day')
            positive = np.bincount(codes, weights=np.maximum(pnl, 0), minlength=len(dates))
            negative = np.bincount(codes, weights=np.maximum(-pnl, 0), minlength=len(dates))
        strategies.append(np.column_stack([returns, positive, negative]))
    return strategies, market-rf, rf, ff, dates, source


def _metrics(strategy, market_excess, rf, ff=None):
    r, positive, negative = strategy.T
    excess = r-rf
    x = np.column_stack([np.ones(len(r)), market_excess])
    alpha = np.linalg.lstsq(x, excess, rcond=None)[0][0]*252 if np.linalg.matrix_rank(x) == 2 else np.nan
    sd = excess.std(ddof=1)
    path = np.r_[1., np.cumprod(1+r)]
    out = [np.log1p(r).mean()*252, alpha,
           excess.mean()/sd*np.sqrt(252) if sd > 0 else np.nan,
           np.min(path/np.maximum.accumulate(path)-1),
           positive.sum()/negative.sum() if negative.sum() > 0 else np.nan,
           path[-1]-1]
    if ff is not None:
        xx = np.column_stack([np.ones(len(r)), ff])
        out.append(np.linalg.lstsq(xx, excess, rcond=None)[0][0]*252
                   if np.linalg.matrix_rank(xx) == xx.shape[1] else np.nan)
    return np.asarray(out)


def _economic_bootstrap(results, index_prices, *, factors=None, n_boot=2000,
                        block=21, seed=42, alpha=.05):
    if n_boot < 1:
        raise ValueError('Positive n_boot required')
    strategies, market, rf, ff, dates, source = _economic_inputs(results, index_prices, factors)
    names = list(METRICS)+(['ff_alpha'] if ff is not None else [])
    estimates = [_metrics(s, market, rf, ff) for s in strategies]
    observed = estimates[0] if len(results) == 1 else estimates[0]-estimates[1]
    draws = np.empty((n_boot, len(names)))
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        ix = circular_blocks(len(dates), block, rng)
        values = [_metrics(s[ix], market[ix], rf[ix], ff[ix] if ff is not None else None)
                  for s in strategies]
        draws[i] = values[0] if len(results) == 1 else values[0]-values[1]
    rows = []
    for j, name in enumerate(names):
        null = 1. if len(results) == 1 and name == 'profit_factor' else 0.
        inference = bootstrap_inference(observed[j], draws[:, j], null=null, alpha=alpha)
        rows.append(dict(metric=name, estimate=observed[j], null=null, block=block,
                         n_days=len(dates), capm_source=source,
                         estimand=('conditional_observed_notional_realized_trade_PnL'
                                   if name == 'profit_factor' else 'fixed_horizon_secondary'
                                   if name in ('total_return', 'max_drawdown') else 'daily_process'),
                         **inference))
    out = pd.DataFrame(rows)
    # Keep attrs scalar/JSON-compatible so independently generated tables can
    # be concatenated by pandas without ambiguous ndarray equality.
    out.attrs.update(seed=seed, date_start=str(dates[0]), date_end=str(dates[-1]),
                     limitation='Conditional on fitted forecasts and observed notional/trade path; no strategy regeneration.',
                     factors_units='decimal daily returns', paired=len(results) == 2)
    return out


def economic_bootstrap(res_a, res_b, index_prices, *, factors=None, n_boot=2000,
                       block=21, seed=42, alpha=.05):
    """Paired contrasts with joint date blocks for returns, RF, factors and PnL."""
    return _economic_bootstrap([res_a, res_b], index_prices, factors=factors,
                               n_boot=n_boot, block=block, seed=seed, alpha=alpha)


def individual_bootstrap(res, index_prices, *, factors=None, n_boot=2000,
                         block=21, seed=42, alpha=.05):
    """Individual tests against zero (profit factor against one)."""
    return _economic_bootstrap([res], index_prices, factors=factors,
                               n_boot=n_boot, block=block, seed=seed, alpha=alpha)

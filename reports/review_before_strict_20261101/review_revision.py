"""Auditable helpers for the revised notebook. No data downloads on import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

K_VALUES = [1, 5, 10, 15, 20, 40, 60, 80, 100, 120, 150, 180, 260]
BASE_FEATURES = [f'X_{k}' for k in K_VALUES]
EXTRA_FEATURES = [f'X_{k}_{stat}_{w}' for k in K_VALUES
                  for w in [2, 5, 10, 20] for stat in ['mean','std','min','max','cv']]
ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download_universe(symbols, cache_dir, start='2014-12-22', end='2024-12-21', retries=2):
    """Frozen original symbols; no automatic replacement with successor companies.

    Explicit unadjusted Close preserves the old price-return definition.
    Yahoo Close is split-adjusted, not dividend-adjusted. No mixing with Adj Close.
    Individual files are reusable only with a matching request manifest.
    """
    import time
    import yfinance as yf
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir / 'yf_metadata'))
    requested = sorted(set(symbols))
    request = dict(start=start, end=end, auto_adjust=False, price='Close',
                   symbols=requested, index='^GSPC')
    request_file = cache_dir / 'request.json'
    if request_file.exists() and json.loads(request_file.read_text()) != request:
        raise ValueError('Cache request differs: choose a new cache directory')
    request_file.write_text(json.dumps(request, indent=2), encoding='utf-8')
    frames, audit = [], []
    for j, symbol in enumerate(requested + ['^GSPC']):
        yahoo = symbol.replace('.', '-')
        path = cache_dir / f'{yahoo.replace("^", "INDEX_")}.csv'
        error = ''
        frame = None
        if path.exists():
            frame = pd.read_csv(path, parse_dates=['Date'])
        else:
            for attempt in range(retries):
                try:
                    raw = yf.download(yahoo, start=start, end=end, interval='1d',
                                      auto_adjust=False, back_adjust=False, actions=False,
                                      repair=False, keepna=True, progress=False,
                                      threads=False, timeout=20, multi_level_index=False)
                    if raw is None or raw.empty:
                        raise ValueError('Yahoo returned no history')
                    close = raw['Close']
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    frame = close.rename('Close').rename_axis('Date').reset_index()
                    frame['Date'] = pd.to_datetime(frame.Date).dt.tz_localize(None).dt.normalize()
                    frame = frame.dropna(subset=['Close'])
                    frame = frame[(frame.Close > 0) & np.isfinite(frame.Close)]
                    if frame.empty:
                        raise ValueError('No finite positive Close')
                    frame.to_csv(path, index=False)
                    break
                except Exception as exc:
                    error = str(exc)
                    frame = None
                    if attempt + 1 < retries:
                        time.sleep(1)
        ok = frame is not None and not frame.empty
        audit.append(dict(Symbol=symbol, YahooSymbol=yahoo, status='ok' if ok else 'unavailable',
                          n_quotes=len(frame) if ok else 0,
                          first=str(frame.Date.min().date()) if ok else '',
                          last=str(frame.Date.max().date()) if ok else '',
                          sha256=sha256(path) if ok else '', error='' if ok else error))
        if ok:
            frames.append(frame.assign(Symbol=symbol))
        # Save progress even if the process is interrupted.
        pd.DataFrame(audit).to_csv(cache_dir / 'download_manifest.csv', index=False)
        if (j+1) % 25 == 0 or not ok:
            print(f'yfinance {j+1}/{len(requested)+1}: {symbol} ' + ('ok' if ok else error), flush=True)
    audit = pd.DataFrame(audit)
    if not frames:
        raise RuntimeError('No Yahoo histories available; see download_manifest.csv')
    full = pd.concat(frames, ignore_index=True)
    idx = full[full.Symbol == '^GSPC'][['Date','Close']].rename(columns={'Close':'SP500'})
    if idx.empty:
        raise RuntimeError('^GSPC unavailable; no silent fallback to another index')
    return full[full.Symbol != '^GSPC'].copy(), idx, audit


def build_base_panel(prices, index, horizon=10):
    """Common trading calendar. Missing sessions remain missing; no bfill/ffill.

    Require ALL future closes for an observed label, including early positives.
    LabelEnd is retained so each split can assert information availability.
    """
    idx = index[['Date','SP500']].sort_values('Date').drop_duplicates('Date').set_index('Date')
    parts = []
    for symbol, group in prices.groupby('Symbol', sort=True):
        group = group.sort_values('Date').set_index('Date')
        if group.index.has_duplicates:
            raise ValueError(f'Duplicate quotes: {symbol}')
        # Include the entire common calendar; IPO dates are never backfilled.
        out = group[['Close']].reindex(idx.index).join(idx)
        out['Symbol'] = symbol
        for k in K_VALUES:
            out[f'X_{k}'] = (out.Close / out.Close.shift(k) - out.SP500 / out.SP500.shift(k))*100
        future = pd.concat([(out.Close.shift(-k)/out.Close -
                             out.SP500.shift(-k)/out.SP500)*100
                            for k in range(1,horizon+1)], axis=1)
        valid = future.notna().all(axis=1) & out.Close.notna() & out.SP500.notna()
        out['Target'] = (future.max(axis=1) > 2).astype(float).where(valid)
        out['LabelEnd'] = pd.Series(idx.index,index=idx.index).shift(-horizon)
        parts.append(out.reset_index())
    return pd.concat(parts, ignore_index=True)


def add_statistical_features(panel):
    parts = []
    for _, group in panel.groupby('Symbol',sort=True):
        group = group.sort_values('Date').reset_index(drop=True)
        features = {}
        for k in K_VALUES:
            x = group[f'X_{k}']
            for w in [2,5,10,20]:
                roll = x.rolling(w,min_periods=1)
                for stat in ['mean','std','min','max']:
                    features[f'X_{k}_{stat}_{w}'] = getattr(roll,stat)()
                features[f'X_{k}_cv_{w}'] = roll.std() / (x.abs().rolling(w,min_periods=1).mean()+1e-6)
        parts.append(pd.concat([group,pd.DataFrame(features)],axis=1))
    return pd.concat(parts,ignore_index=True)


def date_cv_splits(frame, n_splits=5, gap=10):
    dates = np.sort(frame.Date.unique())
    result = []
    for tr,te in TimeSeriesSplit(n_splits=n_splits,gap=gap).split(dates):
        train = np.flatnonzero(frame.Date.isin(dates[tr]).to_numpy() &
                               (frame.LabelEnd < dates[te[0]]).to_numpy())
        test = np.flatnonzero(frame.Date.isin(dates[te]).to_numpy())
        if not len(train) or not len(test):
            raise ValueError('Empty date CV fold')
        result.append((train,test))
    return result


def walk_plan(frame, start, end, *, window=None, gap=10, step=21):
    dates = np.sort(frame.Date.unique())
    test_dates = dates[(dates >= np.datetime64(start)) & (dates <= np.datetime64(end))]
    plans = []
    for pos in range(0,len(test_dates),step):
        first = test_dates[pos]
        calendar_pos = np.searchsorted(dates,first)
        history = dates[:max(0,calendar_pos-gap)]
        if window is not None:
            history = history[-window:]
        if len(history) < (window or 1):
            raise ValueError('Insufficient training history for scheduled fold')
        train = np.flatnonzero(frame.Date.isin(history).to_numpy() &
                               (frame.LabelEnd < first).to_numpy() & frame.Target.notna().to_numpy())
        test = np.flatnonzero(frame.Date.isin(test_dates[pos:pos+step]).to_numpy())
        plans.append(dict(fold=len(plans)+1,train=train,test=test))
    return plans


def walk_predict(frame, features, models, start, end, *, window=None, gap=10, step=21,
                 static=False, cache_dir=None):
    """Same OOS rows for each model. Static freezes the FIRST scheduled fit."""
    plans = walk_plan(frame,start,end,window=window,gap=gap,step=step)
    import sklearn
    cache_protocol = (sha256(Path(__file__))+sklearn.__version__+np.__version__+pd.__version__).encode()
    frames, scores, fitted = [], [], {}
    for plan in plans:
        tr,te = frame.iloc[plan['train']], frame.iloc[plan['test']]
        if tr.Target.nunique() != 2:
            raise ValueError('Training fold has one class')
        if tr.LabelEnd.max() >= te.Date.min():
            raise AssertionError('Future label leakage')
        out = te[['Date','Symbol','Target']].rename(columns={'Target':'y_true'}).copy()
        out['fold'] = plan['fold']
        for name,template in models.items():
            # Cache key includes exact data and estimator: no stale predictions.
            fit_tr = frame.iloc[plans[0]['train']] if static else tr
            payload = pd.concat([fit_tr[['Date','Symbol','Target','LabelEnd']+features],
                                 te[['Date','Symbol']+features]],ignore_index=True)
            digest = hashlib.sha256(pd.util.hash_pandas_object(payload,index=False).values.tobytes()
                                    + repr(template.get_params()).encode()+cache_protocol).hexdigest()[:24]
            path = Path(cache_dir)/f'{name.replace(" ","_")}_{digest}.csv' if cache_dir else None
            if path is not None and path.exists():
                cached = pd.read_csv(path,parse_dates=['Date'])
                if not cached[['Date','Symbol']].reset_index(drop=True).equals(out[['Date','Symbol']].reset_index(drop=True)):
                    raise ValueError('Prediction cache keys differ')
                prob = cached.y_prob.to_numpy()
            else:
                if not static or name not in fitted:
                    fitted[name] = clone(template).fit(fit_tr[features],fit_tr.Target.astype(int))
                prob = fitted[name].predict_proba(te[features])[:,1]
                if path is not None:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    out[['Date','Symbol']].assign(y_prob=prob).to_csv(path,index=False)
            out[name] = prob
            observed = out.y_true.notna().to_numpy()
            auc = roc_auc_score(out.y_true[observed],prob[observed]) if out.y_true.nunique()==2 else np.nan
            scores.append(dict(fold=plan['fold'],Модель=name,**{'ROC-AUC':auc},
                               n_train=len(fit_tr),n_test=len(te),train_end=fit_tr.Date.max(),
                               label_end=fit_tr.LabelEnd.max(),test_start=te.Date.min(),test_end=te.Date.max()))
        frames.append(out)
        print(f'fold {plan["fold"]}/{len(plans)} train={len(tr):,} test={len(te):,}',flush=True)
    return pd.concat(frames,ignore_index=True),pd.DataFrame(scores)


def bh_adjust(pvalues):
    p = np.asarray(pvalues,dtype=float)
    order = np.argsort(p)
    ranked = p[order]*len(p)/np.arange(1,len(p)+1)
    result = np.empty(len(p))
    result[order] = np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1])
    return result


def circular_blocks(n, block, rng):
    if n < 1 or block < 1:
        raise ValueError('Positive n and block required')
    starts = rng.integers(0,n,size=int(np.ceil(n/block)))
    return ((starts[:,None]+np.arange(block)) % n).ravel()[:n]


def risk_metrics(equity, index, *, lag=10):
    """Daily CAPM with rf=0 and Newey-West HAC; alpha is arithmetic 252*a.

    This is a zero-rate price-return CAPM approximation, not FF multi-factor alpha.
    """
    stock = equity.Capital.pct_change().dropna()
    benchmark = index.set_index('Date').SP500.reindex(equity.index).pct_change().dropna()
    aligned = pd.concat([stock.rename('r'),benchmark.rename('m')],axis=1).dropna()
    if len(aligned)<4:
        return dict(jensen_alpha=np.nan,jensen_p=np.nan,beta=np.nan,information_ratio=np.nan)
    y,m = aligned.r.to_numpy(),aligned.m.to_numpy()
    x = np.column_stack([np.ones(len(y)),m])
    coef = np.linalg.lstsq(x,y,rcond=None)[0]
    resid = y-x@coef
    xu = x*resid[:,None]
    meat = xu.T@xu
    for k in range(1,min(lag,len(y)-1)+1):
        gamma = xu[k:].T@xu[:-k]
        meat += (1-k/(lag+1))*(gamma+gamma.T)
    inv = np.linalg.pinv(x.T@x)
    cov = inv@meat@inv
    se = np.sqrt(max(0,cov[0,0]))
    active = y-m
    return dict(jensen_alpha=coef[0]*252,
                jensen_p=float(2*norm.sf(abs(coef[0]/se))) if se>0 else np.nan,
                beta=coef[1], information_ratio=active.mean()/active.std(ddof=1)*np.sqrt(252)
                if active.std(ddof=1)>0 else np.nan)


def _backtest(preds,panel,index_prices,*,horizon,start,end,top_k=10,commission=.001,
              capital=100000,long_short=False,winsorize=None,label='',offset=0):
    if horizon<1 or top_k<1 or commission<0:
        raise ValueError('Invalid strategy parameters')
    if winsorize is not None:
        raise ValueError('Realized returns must not be winsorized using OOS quantiles')
    if preds.duplicated(['Date','Symbol']).any():
        raise ValueError('Duplicate prediction keys')
    idx = index_prices.sort_values('Date').drop_duplicates('Date')
    dates = pd.DatetimeIndex(idx[(idx.Date>=start)&(idx.Date<=end)].Date)
    if len(dates)<3:
        raise ValueError('Insufficient evaluation dates')
    px = panel.pivot(index='Date',columns='Symbol',values='Close').reindex(dates)
    signals = {pd.Timestamp(d):g.sort_values(['y_prob','Symbol'],ascending=[False,True])
               for d,g in preds.groupby('Date')}
    # signal s -> buy at s+1 close -> exit at s+1+h close. Full periods only.
    entries = {s+1:s for s in range(offset,len(dates)-1-horizon,horizon)}
    cash = float(capital)
    holdings = {}
    open_trades = []
    trades, values, periods, turnover = [], [], [], []
    exit_pos = None
    start_value = None
    buy_notional = 0
    for pos,date in enumerate(dates):
        marked = {}
        for symbol,shares in holdings.items():
            price = px.at[date,symbol]
            if not np.isfinite(price) or price<=0:
                raise ValueError(f'Held quote missing: {symbol} {date}; no future-based exclusion')
            marked[symbol] = shares*price
        if holdings and pos==exit_pos:
            sells = sum(abs(v) for v in marked.values())
            cash += sum(marked.values())-commission*sells
            for trade in open_trades:
                exit_price = px.at[date,trade['Symbol']]
                net = trade['Shares']*(exit_price-trade['EntryPrice'])-commission*(abs(trade['Shares'])*exit_price+trade['Notional'])
                trades.append({**trade,'Date':trade['SignalDate'],'ExitDate':date,'ExitPrice':exit_price,
                               'Gross return':trade['SideSign']*(exit_price/trade['EntryPrice']-1),
                               'Net PnL ($)':net})
            periods.append((date,cash/start_value-1))
            turnover.append((buy_notional+sells)/start_value)
            holdings,marked,open_trades = {},{},[]
        if pos in entries:
            signal_date = dates[entries[pos]]
            day = signals.get(signal_date)
            if day is not None:
                # Only entry-time observability determines eligibility.
                day = day[day.Symbol.isin(px.columns)]
                day = day[day.Symbol.map(px.loc[date]).gt(0)]
            if day is not None and len(day)>=top_k*(2 if long_short else 1):
                start_value = cash
                legs = [(day.head(top_k),1)]
                if long_short:
                    legs.append((day.tail(top_k),-1))
                gross = cash/(1+commission)
                allocation = gross/(top_k*len(legs))
                buy_notional = gross
                cash -= commission*gross
                for leg,side in legs:
                    for row in leg.itertuples():
                        price = px.at[date,row.Symbol]
                        shares = side*allocation/price
                        holdings[row.Symbol] = shares
                        cash -= shares*price
                        marked[row.Symbol] = shares*price
                        open_trades.append(dict(SignalDate=signal_date,EntryDate=date,Symbol=row.Symbol,
                                                EntryPrice=price,Shares=shares,Notional=allocation,
                                                Side='long' if side==1 else 'short',SideSign=side,
                                                Prob=row.y_prob,y_true=getattr(row,'y_true',np.nan)))
                exit_pos = pos+horizon
        values.append((date,cash+sum(marked.values())))
        if values[-1][1]<=0:
            raise ValueError('Insolvent strategy: stop instead of compounding negative wealth')
    equity = pd.DataFrame(values,columns=['Date','Capital']).set_index('Date')
    trades = pd.DataFrame(trades)
    daily = equity.Capital.pct_change().dropna()
    dd = equity.Capital/equity.Capital.cummax()-1
    total = equity.Capital.iloc[-1]/capital-1
    years = (dates[-1]-dates[0]).days/365.25
    index_ret = idx.set_index('Date').SP500.loc[dates[-1]]/idx.set_index('Date').SP500.loc[dates[0]]-1
    # Fixed universe observable at first close, identical endpoints for all stocks.
    eligible = px.columns[px.iloc[0].notna() & px.iloc[0].gt(0)]
    ew = np.nan
    if len(eligible) and px[eligible].notna().all().all():
        ew = (px[eligible].iloc[-1]/px[eligible].iloc[0]-1).mean()
    pnl = trades['Net PnL ($)'] if len(trades) else pd.Series(dtype=float)
    losses = pnl[pnl<0].sum()
    metrics = dict(start=dates[0],end=dates[-1],years=years,final_capital=equity.Capital.iloc[-1],
                   total_return=total,annual_return=(1+total)**(1/years)-1,
                   sharpe=daily.mean()/daily.std(ddof=1)*np.sqrt(252) if daily.std(ddof=1)>0 else np.nan,
                   max_drawdown=dd.min(),n_trades=len(trades),win_rate=(pnl>0).mean(),
                   avg_win=pnl[pnl>0].mean(),avg_loss=pnl[pnl<0].mean(),
                   profit_factor=pnl[pnl>0].sum()/abs(losses) if losses<0 else np.nan,
                   bh_index=index_ret,bh_equal_weight=ew,alpha_vs_index=total-index_ret,
                   alpha_vs_equal_weight=total-ew,n_rebalances=len(periods),
                   avg_turnover=float(np.mean(turnover)) if turnover else 0,
                   one_way_turnover=float(np.mean(turnover))/2 if turnover else 0)
    metrics.update(risk_metrics(equity,idx,lag=max(10,horizon)))
    return dict(label=label,equity=equity,trades=trades,drawdown=dd,preds=preds,
                period_returns=pd.Series(dict(periods),dtype=float),
                params=dict(horizon=horizon,top_k=top_k,commission=commission,capital=capital,
                            long_short=long_short,winsorize=None),metrics=metrics)


def run_backtest(preds,panel,index_prices,**kwargs):
    return _backtest(preds,panel,index_prices,**kwargs)


def overlapping_backtest(preds,panel,index_prices,*,horizon,capital=100000,**kwargs):
    """h staggered sleeves (1/h capital each), launched on successive sessions.

    Each sleeve holds h days, with daily aggregate mark-to-market. At the fixed
    endpoint no new incomplete positions are opened. Early/late unused cash
    remains cash for BOTH horizons; boundary effects are explicit.
    """
    sleeves = [_backtest(preds,panel,index_prices,horizon=horizon,capital=capital/horizon,
                         offset=offset,**kwargs) for offset in range(horizon)]
    out = sleeves[0].copy()
    equity = sum(s['equity'].Capital for s in sleeves).to_frame('Capital')
    daily = equity.Capital.pct_change().dropna()
    dd = equity.Capital/equity.Capital.cummax()-1
    trades = pd.concat([s['trades'].assign(Sleeve=i) for i,s in enumerate(sleeves)],ignore_index=True)
    total = equity.Capital.iloc[-1]/capital-1
    m = dict(out['metrics'])
    m.update(final_capital=equity.Capital.iloc[-1],total_return=total,
             annual_return=(1+total)**(1/m['years'])-1,max_drawdown=dd.min(),
             sharpe=daily.mean()/daily.std(ddof=1)*np.sqrt(252) if daily.std(ddof=1)>0 else np.nan,
             alpha_vs_index=total-m['bh_index'],alpha_vs_equal_weight=total-m['bh_equal_weight'],
             n_trades=len(trades),n_rebalances=sum(s['metrics']['n_rebalances'] for s in sleeves))
    pnl = trades['Net PnL ($)'] if len(trades) else pd.Series(dtype=float)
    loss = pnl[pnl<0].sum()
    m.update(win_rate=(pnl>0).mean(),profit_factor=pnl[pnl>0].sum()/abs(loss) if loss<0 else np.nan,
             avg_win=pnl[pnl>0].mean(),avg_loss=pnl[pnl<0].mean())
    # Aggregate traded notional over mean aggregate wealth per evaluation session.
    notional = sum((s['trades'].Notional + (s['trades'].Shares.abs()*s['trades'].ExitPrice)).sum()
                   for s in sleeves if len(s['trades']))
    m['avg_turnover'] = notional/equity.Capital.mean()/max(1,len(equity)-1)
    m['one_way_turnover'] = m['avg_turnover']/2
    m.update(risk_metrics(equity,index_prices,lag=max(10,horizon)))
    out.update(equity=equity,drawdown=dd,trades=trades,metrics=m,
               params={**out['params'],'capital':capital,'overlapping':True})
    return out


def paired_daily_bootstrap(a,b,n_boot=2000,block=21,seed=42):
    """Paired circular blocks on SAME daily dates; centered test, plus-one p."""
    ra = a['equity'].Capital.pct_change().dropna()
    rb = b['equity'].Capital.pct_change().dropna()
    if not ra.index.equals(rb.index):
        raise ValueError('Daily evaluation calendars differ')
    av,bv = ra.to_numpy(),rb.to_numpy()
    observed = np.prod(1+av)-np.prod(1+bv)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        ix = circular_blocks(len(av),block,rng)
        draws[i] = np.prod(1+av[ix])-np.prod(1+bv[ix])
    lo,hi = np.quantile(draws,[.025,.975])
    p = (1+np.sum(np.abs(draws-observed)>=abs(observed)))/(n_boot+1)
    return dict(diff=observed,ci_low=lo,ci_high=hi,p=p,n_periods=len(av),n_boot=n_boot,block=block)

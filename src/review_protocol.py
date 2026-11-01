"""Preflight, artifact persistence, and post-run integrity checks."""
import importlib.metadata
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def preflight(directory):
    required=['numpy','pandas','scipy','sklearn','matplotlib','seaborn','joblib','xgboost','shap','flaml']
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Missing packages in {sys.executable}: {missing}. Install BEFORE training.')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    environment=dict(python=sys.version,executable=sys.executable,packages={})
    root=Path(__file__).resolve().parents[1]
    environment['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root/'src').glob('review_*.py'))}
    for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','seaborn','joblib','xgboost','shap','flaml','yfinance','ipykernel']:
        try:
            environment['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment['packages'][name]='not-installed'
    # Exercise actual compressed persistence now; no optional parquet dependency.
    path=save_predictions(pd.DataFrame({'Date':[pd.Timestamp('2000-01-01')],'value':[1.]}),directory/'preflight_write')
    restored=pd.read_csv(path,parse_dates=['Date'])
    if len(restored)!=1 or restored.value.iloc[0]!=1:
        raise RuntimeError('Prediction persistence roundtrip failed')
    (directory/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    return environment


def save_predictions(frame,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path=path.with_name(path.name+'.csv.gz')
    frame.to_csv(path,index=False,compression='gzip',float_format='%.17g')
    return path


def validate_backtest(result,start,end):
    equity=result['equity']
    assert not equity.index.has_duplicates,'duplicate equity dates'
    assert equity.index.is_monotonic_increasing,'equity order'
    assert equity.index[0]==pd.Timestamp(start),'equity start'
    assert equity.index[-1]==pd.Timestamp(end),'equity end'
    assert np.isfinite(equity.Capital).all() and equity.Capital.gt(0).all(),'invalid equity'
    assert np.isclose(equity.Capital.iloc[0],result['params']['capital']),'initial capital missing'
    trades=result['trades']
    if len(trades):
        assert trades.ExitDate.le(end).all(),'trade exit beyond evaluation'
        assert trades.EntryDate.gt(trades.SignalDate).all(),'same-close execution'
        assert trades.ExitDate.gt(trades.EntryDate).all(),'exit before entry'
        assert np.isclose(trades['Net PnL ($)'].sum(),equity.Capital.iloc[-1]-equity.Capital.iloc[0],rtol=1e-7,atol=1e-5),'PnL does not reconcile'
    return True


def validate_inference(table):
    ok=table.status.eq('ok')
    data=table.loc[ok]
    null=data['null'] if 'null' in data else 0.
    excludes=(data.ci_low>null)|(data.ci_high<null)
    assert (excludes.to_numpy()==data.p.lt(data.alpha).to_numpy()).all(),'CI/p inversion inconsistent'
    assert data.p.gt(0).all() and data.p.le(1).all(),'invalid bootstrap p'
    if 'p_Holm' in data:
        assert (data.p_Holm>=data.p-1e-14).all(),'Holm decreased a p-value'
    return True


def validate_predictions(predictions):
    assert not predictions.duplicated(['Date','Symbol']).any(),'duplicate prediction keys'
    for name in predictions.columns.difference(['Date','Symbol','y_true','fold']):
        values=predictions[name]
        assert np.isfinite(values).all() and values.between(0,1).all(),f'invalid probabilities {name}'
    return True

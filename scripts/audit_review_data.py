"""Audit downloaded histories without fitting models or claiming full coverage."""
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import review_revision as rr
cache=ROOT/'data/versions/yfinance_review_2026_11_01'
out=ROOT/'data/derived/review_2026_11_01'
out.mkdir(parents=True,exist_ok=True)
audit=pd.read_csv(cache/'download_manifest.csv')
index=pd.read_csv(cache/'INDEX_GSPC.csv',parse_dates=['Date']).rename(columns={'Close':'SP500'})
rows=[]
for record in audit[audit.Symbol!='^GSPC'].itertuples():
    if record.status!='ok':
        rows.append(dict(Symbol=record.Symbol,status=record.status,quotes=0,feature_rows=0,labeled_rows=0,oos_rows=0,positives=0))
        continue
    prices=pd.read_csv(cache/(record.YahooSymbol+'.csv'),parse_dates=['Date']).assign(Symbol=record.Symbol)
    df=rr.add_statistical_features(rr.build_base_panel(prices,index))
    ready=df[rr.ALL_FEATURES].notna().all(axis=1)
    labeled=ready & df.Target.notna()
    oos=ready & df.Date.between('2022-06-02','2024-11-22')
    rows.append(dict(Symbol=record.Symbol,status='ok',quotes=len(prices),feature_rows=int(ready.sum()),
                     labeled_rows=int(labeled.sum()),oos_rows=int(oos.sum()),
                     oos_unknown_target=int((oos & df.Target.isna()).sum()),
                     positives=int(df.loc[labeled,'Target'].sum())))
result=pd.DataFrame(rows)
result.to_csv(out/'downloaded_data_audit.csv',index=False)
summary=dict(requested=int(len(result)),loaded=int(result.status.eq('ok').sum()),
             with_273_features=int(result.feature_rows.gt(0).sum()),
             labeled_symbols=int(result.labeled_rows.gt(0).sum()),
             labeled_rows=int(result.labeled_rows.sum()),
             class1=float(result.positives.sum()/result.labeled_rows.sum()),
             oos_symbols=int(result.oos_rows.gt(0).sum()),oos_prediction_rows=int(result.oos_rows.sum()),
             oos_unknown_targets=int(result.oos_unknown_target.sum()),
             missing_symbols=result.loc[result.status!='ok','Symbol'].tolist(),
             experiment_status='DATA AUDIT ONLY; no full experiment fitted')
(out/'downloaded_data_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

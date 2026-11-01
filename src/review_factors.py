"""Frozen public daily FF5+momentum factors; percentages converted to decimals."""
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

URLS = {
    'ff5': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
    'momentum': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip',
}
FACTOR_COLUMNS=['Mkt-RF','SMB','HML','RMW','CMA','Mom','RF']


def parse_french_daily(text):
    header=None
    rows=[]
    for line in text.splitlines():
        cells=[x.strip() for x in next(csv.reader([line]))]
        if cells and not cells[0] and any(x in cells for x in ['Mkt-RF','Mom']):
            header=['Date']+cells[1:]
        elif cells and re.fullmatch(r'\d{8}',cells[0]) and header is not None:
            if len(cells)!=len(header):
                raise ValueError('Unexpected French factor row width')
            rows.append(cells)
    if not rows:
        raise ValueError('No daily factor rows found')
    frame=pd.DataFrame(rows,columns=header)
    frame['Date']=pd.to_datetime(frame.Date,format='%Y%m%d')
    for name in header[1:]:
        frame[name]=pd.to_numeric(frame[name],errors='raise')
        frame.loc[frame[name].isin([-99.99,-999]),name]=np.nan
        frame[name]=frame[name]/100
    if frame.Date.duplicated().any():
        raise ValueError('Duplicate factor dates')
    return frame


def require_coverage(factors,dates):
    dates=pd.DatetimeIndex(dates)
    aligned=factors.set_index('Date').reindex(dates)
    if not set(FACTOR_COLUMNS).issubset(aligned) or not np.isfinite(aligned[FACTOR_COLUMNS]).all().all():
        missing=dates[aligned.isna().any(axis=1)]
        raise ValueError(f'Factors do not cover requested dates: {list(missing[:5])}')
    return aligned.reset_index(names='Date')


def prepare_factors(directory,*,download=False):
    """Offline by default. A refresh must explicitly request network access."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    inputs=[]
    manifest=[]
    for name,url in URLS.items():
        path=directory/f'{name}.zip'
        if download:
            import requests
            response=requests.get(url,timeout=60)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                archive.testzip()
            path.write_bytes(response.content)
        if not path.exists():
            raise FileNotFoundError(f'Missing frozen factors: {path}. Run scripts/download_review_factors.py before training.')
        raw=path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            member=next(x for x in archive.namelist() if x.lower().endswith('.csv'))
            frame=parse_french_daily(archive.read(member).decode('utf-8-sig'))
        inputs.append(frame)
        manifest.append(dict(name=name,url=url,file=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                             first=str(frame.Date.min().date()),last=str(frame.Date.max().date()),
                             raw_units='percent daily',output_units='decimal daily'))
    joined=inputs[0].merge(inputs[1],on='Date',how='outer',validate='one_to_one').sort_values('Date')
    joined.to_csv(directory/'ff5_momentum_daily.csv',index=False)
    (directory/'factor_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return joined

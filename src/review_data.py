"""Offline, strict recovery of the fixed coursework security universe.

Only refresh=True permits network access. No successor/acquirer substitution,
forward filling, synthetic histories, or dividend-adjusted Close is allowed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_CONVENTION = 'split_adjusted_close_excluding_dividends'
REQUIRED_LABEL_END = '2024-12-09'  # ten market sessions after fixed OOS end 2024-11-22
# Same issuer and security; these are ticker changes, not acquisitions.
TICKER_ALIASES = {'BK': 'BNY', 'FI': 'FISV', 'MMC': 'MRSH'}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _quotes(path, dates, symbol):
    frame = pd.read_csv(path, usecols=['Date', 'Close'], parse_dates=['Date'])
    frame['Date'] = pd.to_datetime(frame.Date).dt.tz_localize(None).dt.normalize()
    frame['Close'] = pd.to_numeric(frame.Close, errors='raise')
    frame = frame.loc[frame.Date.isin(dates) & frame.Close.gt(0) & np.isfinite(frame.Close)]
    if frame.Date.duplicated().any():
        raise ValueError(f'Duplicate quote dates for {symbol}: {path}')
    return frame.sort_values('Date').reset_index(drop=True)


def prepare_review_data(data_dir, yf_dir, output_dir, require_full=True, refresh=False):
    """Return (Date/Symbol/Close prices, Date/SP500 index, per-security audit).

    The universe comes exclusively from sp500_companies.csv. A complete load
    writes a frozen SHA256 manifest; subsequent offline loads reject changes.
    Incomplete loads always write the audit and missing-history import template,
    and raise by default. require_full=False is for diagnostics only.

    Optional manual imports live in data_dir/review_manual_quotes/manifest.json,
    keyed by original symbol with file, security_symbol, source_url,
    price_convention, and optional notes. Their Date,Close CSV must represent
    that exact security and PRICE_CONVENTION; normalization must be documented
    before import. An assertion in metadata is not a replacement for review.
    """
    data_dir, yf_dir, output_dir = map(Path, (data_dir, yf_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    companies_path = data_dir / 'sp500_companies.csv'
    symbols = sorted(pd.read_csv(companies_path).Symbol.dropna().unique().tolist())
    if not symbols:
        raise ValueError('Original security universe is empty')
    if refresh:
        from src.review_revision import download_universe
        download_universe(symbols, yf_dir)
        download_universe(list(TICKER_ALIASES.values()), yf_dir.with_name(yf_dir.name + '_aliases'))
    request_path = yf_dir / 'request.json'
    request = json.loads(request_path.read_text(encoding='utf-8'))
    if (sorted(request['symbols']) != symbols or request.get('auto_adjust') is not False
            or request.get('price') != 'Close' or request.get('index') != '^GSPC'):
        raise ValueError('Yahoo request does not match fixed universe / Close convention')
    index_path = yf_dir / 'INDEX_GSPC.csv'
    index_raw = pd.read_csv(index_path, usecols=['Date', 'Close'], parse_dates=['Date'])
    index_raw = index_raw.loc[index_raw.Date.ge(request['start']) & index_raw.Date.lt(request['end'])]
    if index_raw.empty or index_raw.Date.duplicated().any() or not (np.isfinite(index_raw.Close) & index_raw.Close.gt(0)).all():
        raise ValueError('Invalid ^GSPC history; no substitute index allowed')
    index = index_raw.sort_values('Date').rename(columns={'Close': 'SP500'}).reset_index(drop=True)
    dates = index.Date
    source_hashes = {}

    def record(path):
        source_hashes[str(path.resolve())] = _hash(path)

    for path in (companies_path, request_path, index_path):
        record(path)

    def verify_auxiliary_cache(directory):
        auxiliary_request = directory / 'request.json'
        if not auxiliary_request.exists():
            raise ValueError(f'Unverified cache price convention (request missing): {directory}')
        metadata = json.loads(auxiliary_request.read_text(encoding='utf-8'))
        if (metadata.get('auto_adjust') is not False or metadata.get('price') != 'Close'
                or metadata.get('index') != '^GSPC'):
            raise ValueError(f'Unverified cache price convention: {directory}')
        record(auxiliary_request)
    manual_dir = data_dir / 'review_manual_quotes'
    manual_path = manual_dir / 'manifest.json'
    manual = json.loads(manual_path.read_text(encoding='utf-8')) if manual_path.exists() else {}
    if manual_path.exists():
        record(manual_path)
    legacy_path = data_dir / 'sp500_stocks.csv'
    legacy = None
    frames, rows = [], []
    for symbol in symbols:
        frame, source, yahoo, source_path, source_url = pd.DataFrame(), '', '', None, ''
        for directory in (yf_dir, yf_dir.with_name(yf_dir.name + '_recovery')):
            path = directory / (symbol.replace('.', '-') + '.csv')
            if path.exists():
                if directory != yf_dir:
                    verify_auxiliary_cache(directory)
                frame = _quotes(path, dates, symbol)
                if not frame.empty:
                    source, yahoo, source_path = 'Yahoo Close', symbol.replace('.', '-'), path
                    break
        if frame.empty and symbol in TICKER_ALIASES:
            alias = TICKER_ALIASES[symbol]
            path = yf_dir.with_name(yf_dir.name + '_aliases') / f'{alias}.csv'
            if path.exists():
                verify_auxiliary_cache(path.parent)
                frame = _quotes(path, dates, symbol)
                if not frame.empty:
                    source, yahoo, source_path = 'Yahoo verified same-security ticker change', alias, path
        if frame.empty and legacy_path.exists():
            if legacy is None:
                legacy = pd.read_csv(legacy_path, usecols=['Date', 'Symbol', 'Close'], parse_dates=['Date'])
                legacy['Close'] = pd.to_numeric(legacy.Close, errors='raise')
            frame = legacy.loc[legacy.Symbol.eq(symbol) & legacy.Date.isin(dates)
                               & legacy.Close.gt(0) & np.isfinite(legacy.Close), ['Date', 'Close']].sort_values('Date')
            if frame.Date.duplicated().any():
                raise ValueError(f'Duplicate legacy quotes: {symbol}')
            if not frame.empty:
                source, source_path = 'Original coursework CSV: Close (not Adj Close)', legacy_path
        if frame.empty and symbol in manual:
            entry = manual[symbol]
            if entry.get('security_symbol') != symbol:
                raise ValueError(f'Manual security identity mismatch: {symbol}')
            if entry.get('price_convention') != PRICE_CONVENTION:
                raise ValueError(f'Unverified manual price convention for {symbol}')
            if not entry.get('source_url', '').startswith(('https://', 'http://')):
                raise ValueError(f'Manual source URL required: {symbol}')
            path = (manual_dir / entry['file']).resolve()
            if not path.is_relative_to(manual_dir.resolve()):
                raise ValueError(f'Manual CSV escapes import directory: {symbol}')
            frame = _quotes(path, dates, symbol)
            if not frame.empty:
                source, source_path, source_url = 'Reviewed manual import', path, entry['source_url']
        ok = not frame.empty
        if ok:
            record(source_path)
            frames.append(frame.assign(Symbol=symbol)[['Date', 'Symbol', 'Close']])
        rows.append(dict(Symbol=symbol, YahooSymbol=yahoo, status='ok' if ok else 'unavailable',
                         Source=source, SourceURL=source_url, PriceConvention=PRICE_CONVENTION if ok else '',
                         n_quotes=len(frame), first=str(frame.Date.min().date()) if ok else '',
                         last=str(frame.Date.max().date()) if ok else '',
                         required_actual_end=REQUIRED_LABEL_END,
                         covers_required_end=bool(ok and frame.Date.max() >= pd.Timestamp(REQUIRED_LABEL_END)),
                         missing_requested_end_sessions=int(dates.gt(frame.Date.max()).sum()) if ok else len(dates),
                         sha256=source_hashes[str(source_path.resolve())] if ok else '',
                         source_file=str(source_path.resolve()) if ok else '',
                         error='' if ok else 'No verified positive Close history for original security'))
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / 'recovered_download_manifest.csv', index=False)
    missing = audit.loc[audit.status.ne('ok'), 'Symbol'].tolist()
    manifest = dict(schema=1, symbols=symbols, start=request['start'], end_exclusive=request['end'],
                    price_convention=PRICE_CONVENTION, aliases=TICKER_ALIASES,
                    sources=source_hashes, available=len(symbols)-len(missing), missing=missing)
    candidate_path = output_dir / 'review_data_candidate_manifest.json'
    candidate_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    frozen_path = output_dir / 'review_data_manifest.json'
    if missing and not require_full:
        frozen_path = output_dir / 'review_data_partial_manifest.json'
    if frozen_path.exists() and not refresh:
        previous = json.loads(frozen_path.read_text(encoding='utf-8'))
        if previous != manifest:
            raise ValueError('Frozen data manifest differs; use a new output directory or explicit refresh')
    if missing:
        template = {s: dict(file=f'{s}.csv', security_symbol=s, source_url='',
                           price_convention=PRICE_CONVENTION, notes='Supply verified original-security Close, no acquirer substitution') for s in missing}
        (output_dir / 'missing_manual_import_template.json').write_text(json.dumps(template, indent=2), encoding='utf-8')
        if require_full:
            raise RuntimeError(f'Full fixed universe blocked: {len(symbols)-len(missing)}/{len(symbols)} histories; missing: {", ".join(missing)}. See recovered_download_manifest.csv and missing_manual_import_template.json')
        frozen_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    else:
        frozen_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['Date', 'Symbol', 'Close'])
    return prices, index, audit

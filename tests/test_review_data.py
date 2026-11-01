"""Offline checks for strict, auditable recovery of the original universe."""
import json

import pandas as pd
import pytest


def loader():
    # A useful assertion failure, rather than collection failure, before implementation.
    from src import review_data
    return review_data.prepare_review_data


def fixture(tmp_path, symbols):
    data, yf, out = (tmp_path / x for x in ('legacy', 'yahoo', 'out'))
    data.mkdir(); yf.mkdir()
    pd.DataFrame({'Symbol': symbols}).to_csv(data / 'sp500_companies.csv', index=False)
    pd.DataFrame(columns=['Date', 'Symbol', 'Close']).to_csv(data / 'sp500_stocks.csv', index=False)
    (yf / 'request.json').write_text(json.dumps(dict(start='2014-12-22', end='2024-12-21', auto_adjust=False, price='Close', symbols=sorted(symbols), index='^GSPC')))
    quote(yf / 'INDEX_GSPC.csv')
    return data, yf, out


def quote(path, value=100):
    pd.DataFrame({'Date': ['2020-01-02', '2020-01-03'], 'Close': [value, value+1]}).to_csv(path, index=False)


def test_missing_history_is_blocking_and_audit_survives(tmp_path):
    data, yf, out = fixture(tmp_path, ['A', 'MISSING'])
    quote(yf / 'A.csv')
    with pytest.raises(RuntimeError, match='MISSING'):
        loader()(data, yf, out)
    audit = pd.read_csv(out / 'recovered_download_manifest.csv')
    assert audit.loc[audit.Symbol.eq('MISSING'), 'status'].item() == 'unavailable'
    assert not (out / 'review_data_manifest.json').exists()


def test_alias_retains_original_security_and_legacy_uses_close(tmp_path):
    data, yf, out = fixture(tmp_path, ['BK', 'DFS'])
    aliases = yf.with_name(yf.name + '_aliases'); aliases.mkdir()
    (aliases / 'request.json').write_text(json.dumps({'auto_adjust': False, 'price': 'Close', 'index': '^GSPC'}))
    quote(aliases / 'BNY.csv', 50)
    pd.DataFrame({'Date': ['2020-01-02', '2020-01-03'], 'Symbol': ['DFS']*2,
                  'Close': [20, 21], 'Adj Close': [10, 11]}).to_csv(data / 'sp500_stocks.csv', index=False)
    prices, index, audit = loader()(data, yf, out)
    assert set(prices.Symbol) == {'BK', 'DFS'}
    assert prices.loc[prices.Symbol.eq('DFS'), 'Close'].tolist() == [20, 21]
    assert audit.loc[audit.Symbol.eq('BK'), 'YahooSymbol'].item() == 'BNY'
    assert 'split' in audit.loc[audit.Symbol.eq('DFS'), 'PriceConvention'].item()
    assert list(index.columns) == ['Date', 'SP500']


def test_cached_load_never_downloads_and_rejects_changed_inputs(tmp_path, monkeypatch):
    data, yf, out = fixture(tmp_path, ['A'])
    quote(yf / 'A.csv')
    from src import review_revision
    monkeypatch.setattr(review_revision, 'download_universe', lambda *a, **k: pytest.fail('network invoked'))
    loader()(data, yf, out)
    loader()(data, yf, out)
    quote(yf / 'A.csv', 200)
    with pytest.raises(ValueError, match='[Ff]rozen'):
        loader()(data, yf, out)


def test_acquirer_and_dividend_adjusted_manual_history_are_rejected(tmp_path):
    data, yf, out = fixture(tmp_path, ['HES'])
    quote(yf / 'CVX.csv')
    manual = data / 'review_manual_quotes'; manual.mkdir()
    quote(manual / 'HES.csv')
    (manual / 'manifest.json').write_text(json.dumps({'HES': {'file': 'HES.csv', 'security_symbol': 'HES', 'source_url': 'https://example.org/hess', 'price_convention': 'dividend_adjusted'}}))
    with pytest.raises(ValueError, match='convention'):
        loader()(data, yf, out)


def test_partial_mode_is_explicit_and_does_not_freeze_complete_manifest(tmp_path):
    data, yf, out = fixture(tmp_path, ['A', 'HES'])
    quote(yf / 'A.csv')
    prices, _, audit = loader()(data, yf, out, require_full=False)
    assert set(prices.Symbol) == {'A'}
    assert audit.loc[audit.Symbol.eq('HES'), 'status'].item() == 'unavailable'
    assert not (out / 'review_data_manifest.json').exists()


def test_reviewed_manual_close_is_audited_and_partial_snapshot_is_frozen(tmp_path):
    data, yf, out = fixture(tmp_path, ['AVB', 'HES'])
    manual = data / 'review_manual_quotes'; manual.mkdir()
    quote(manual / 'AVB.csv')
    (manual / 'manifest.json').write_text(json.dumps({'AVB': {'file': 'AVB.csv', 'security_symbol': 'AVB', 'source_url': 'https://example.org/original-avb.csv', 'price_convention': 'split_adjusted_close_excluding_dividends'}}))
    prices, _, audit = loader()(data, yf, out, require_full=False)
    row = audit[audit.Symbol.eq('AVB')].iloc[0]
    assert row.Source == 'Reviewed manual import'
    assert row.SourceURL == 'https://example.org/original-avb.csv'
    assert len(row.sha256) == 64
    assert (out / 'review_data_partial_manifest.json').exists()
    quote(manual / 'AVB.csv', 200)
    with pytest.raises(ValueError, match='Frozen'):
        loader()(data, yf, out, require_full=False)


def test_duplicate_quote_dates_are_not_silently_compressed(tmp_path):
    data, yf, out = fixture(tmp_path, ['A'])
    pd.DataFrame({'Date': ['2020-01-02']*2, 'Close': [100, 101]}).to_csv(yf / 'A.csv', index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        loader()(data, yf, out)


def test_alias_cache_cannot_silently_use_adjusted_close(tmp_path):
    data, yf, out = fixture(tmp_path, ['BK'])
    aliases = yf.with_name(yf.name + '_aliases'); aliases.mkdir()
    quote(aliases / 'BNY.csv')
    (aliases / 'request.json').write_text(json.dumps({'auto_adjust': True, 'price': 'Close', 'index': '^GSPC'}))
    with pytest.raises(ValueError, match='convention'):
        loader()(data, yf, out)

# Fixed-universe data recovery, 2026-11-01

The original `sp500_companies.csv` defines **502 securities**. The frozen Yahoo
cache directly supplies 484. Same-security ticker changes supply BK→BNY,
FI→FISV and MMC→MRSH. The original coursework CSV supplies ANSS, DFS, EA,
EQR and K. An archived copy of the same CSV supplies AVB, WBA and WBD.
The resulting panel contains **495 of 502 original securities**. This is a
partial-universe result and must not be described as all 502.

The remaining seven securities are **CTRA, DAY, HES, HOLX, IPG, JNPR, QRVO**.
The original CSV contains no finite Close for them. A fresh Yahoo recovery
request also returned no history, and the Nasdaq historical API returned
`Symbol not exists.` Stooq's public CSV endpoints returned 404, and its Polish
endpoint supplied a browser-verification page. No acquirer prices were used.
Versioned Kaggle download attempts were unavailable (404 or certificate-chain
failure). These failures establish an operational limitation of these attempted
sources, not that the original security histories cannot exist elsewhere.

## Archive admitted into the panel

Source: [SP500-Stock-Prediction, immutable archive commit](https://huggingface.co/khoaliamle/SP500-Stock-Prediction/tree/2fc5a49b09d3d988f26e5c089551780a0c3a1b71/data).
The publisher's API reported commit
`2fc5a49b09d3d988f26e5c089551780a0c3a1b71`, last modified 2025-03-06.
The successfully retrieved `sp500_stocks.csv` contained 103,936,992 bytes.
Only its explicit `Date`, `Symbol`, `Close` columns were read; no `Adj Close`
column was used. Extraction retained the exact AVB, WBA, WBD original symbols,
with no conversion, interpolation, filling, or successor substitution.

Each recovered series contributes 2,509 finite positive quotes on the requested
common calendar, 2014-12-22 through **2024-12-10**. They lack the last eight
requested sessions through 2024-12-20. The fixed OOS evaluation ends 2024-11-22;
its tenth following market session is **2024-12-09**. Consequently these three
series cover the future label tail needed for that evaluation. Their later
missing sessions remain missing. This fact does not justify extending evaluation
past the fixed OOS end without a fresh availability check.

The archived Close column is treated according to the original Yahoo dataset's
split-adjusted, dividend-excluding Close convention, separately from Adj Close.
The full downloaded archive was not retained and its SHA256 was not captured
during the successful retrieval. Later attempts to retrieve it again encountered
a certificate error. The retained extracted CSV files each receive a SHA256 in
the data manifest; the exact source URL and immutable commit are recorded in
their manual-import metadata. There was no independent original-exchange
cross-check of these three archival histories. This provenance limit should
remain visible in the final notebook's limitations.

## Loader and reproducibility

`src.review_data.prepare_review_data(data_dir, yf_dir, output_dir,
require_full=True, refresh=False)` returns stock prices, the original ^GSPC
index and a per-security audit. With `refresh=False`, it performs **no network
requests**. It preserves the original universe identifiers and uses positive,
finite Close only. Duplicate dates cause an error. Request metadata must specify
`auto_adjust=False`, `Close`, the original universe, and ^GSPC.

The audit is always written to `recovered_download_manifest.csv`; it includes
source, price convention, retained input-file SHA256, first/last date, quote
count, and whether the final required label date is covered. `require_full=True`
raises with the exact missing list. Explicit diagnostic/authorized partial mode
uses `require_full=False` and writes `review_data_partial_manifest.json`; a full
universe would write `review_data_manifest.json`. Subsequent cached calls reject
changed inputs under the same frozen output directory. A new snapshot should use
a new output directory (or an explicit refresh).

Artifacts for the final inspected snapshot are in `reports/review_data_recovery_v2/`.
`data/versions/v1.0.0_coursework/review_manual_quotes/` retains the three recovered
CSV series and their source metadata. Additional verified histories can be
supplied through the JSON template written as
`missing_manual_import_template.json`. Every such import must establish the
original security identity and split-adjusted, dividend-excluding convention.
Neither an acquisition nor a matching company name establishes price continuity.

Offline verification: eight loader tests passed; the repository suite passed
37 tests using `.venv/Scripts/python.exe -m pytest -c pytest.ini -q
--basetemp=reports/review_data_fullsuite_tmp2`. Two cached snapshot loads returned
identical prices, index and audit. Auxiliary alias/recovery caches must also
declare unadjusted Close, and their request metadata hashes are frozen.
Bare pytest has a pre-existing
configuration error from duplicate `[tool.ruff.lint.mccabe]` in pyproject.toml.

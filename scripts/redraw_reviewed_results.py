"""Redraw actual completed results from CSV, without fitting or network calls.

In a notebook: %run ../scripts/redraw_reviewed_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'data/derived/review_strict_2026_11_01'
OUTPUT = ROOT / 'reports/review_results_colors'


def redraw():
    completed = json.loads((RESULTS / 'run_completed.json').read_text(encoding='utf-8'))
    assert completed['status'] == 'completed'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    colors = {'static': '#7b3294', 'expanding': '#008837', 'sliding': '#2166ac'}
    labels = {'static': 'Static RF', 'expanding': 'Expanding RF', 'sliding': 'Sliding RF'}
    equity = {
        name: pd.read_csv(RESULTS / f'equity_{name}.csv', parse_dates=['Date'])
        .set_index('Date').Capital for name in colors
    }
    dates = equity['static'].index
    assert all(s.index.equals(dates) for s in equity.values())
    market = pd.read_csv(ROOT / 'data/versions/yfinance_review_2026_11_01/INDEX_GSPC.csv',
                         parse_dates=['Date']).set_index('Date')
    column = 'SP500' if 'SP500' in market else 'Close'
    market = market[column].reindex(dates)
    assert market.notna().all()
    market = market / market.iloc[0] * equity['static'].iloc[0]
    styles = {'static': ':', 'expanding': '-', 'sliding': '-.'}
    with plt.style.context('default'):
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for name, capital in equity.items():
            axes[0].plot(dates, capital, color=colors[name], linestyle=styles[name],
                         linewidth=2, label=labels[name])
            axes[1].plot(dates, capital / capital.cummax() - 1,
                         color=colors[name], linestyle=styles[name], linewidth=1.8,
                         label=labels[name])
        axes[0].plot(dates, market, color='#e66101', linestyle='--', linewidth=2,
                     label='S&P 500 (price)')
        axes[1].plot(dates, market / market.cummax() - 1, color='#e66101',
                     linestyle='--', linewidth=1.8, label='S&P 500 (price)')
        axes[0].set_ylabel('Capital ($)')
        axes[0].set_title('Completed run: RF, holding 10 trading days, costs included')
        axes[1].set_ylabel('Drawdown')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            ax.legend(loc='best')
            ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(OUTPUT / 'capital_and_drawdown.png', dpi=160, bbox_inches='tight')
        fig.savefig(OUTPUT / 'capital_and_drawdown.svg', bbox_inches='tight')
        plt.show()
    return fig


if __name__ == '__main__':
    redraw()

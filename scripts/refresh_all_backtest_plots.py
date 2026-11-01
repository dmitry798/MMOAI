"""Replace all four saved capital/drawdown plots in both reviewed notebooks.

Reuses completed predictions and quotes offline. Does not fit any model.
"""
import base64
import io
import json
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import review_data, review_revision


def main():
    folder = ROOT / 'data/derived/review_strict_2026_11_01'
    output = ROOT / 'reports/review_results_colors'
    output.mkdir(parents=True, exist_ok=True)
    helpers = runpy.run_path(str(ROOT / 'scripts/restore_backtest_colors.py'),
                             run_name='palette_helpers')
    quotes, index_prices, _ = review_data.prepare_review_data(
        ROOT / 'data/versions/v1.0.0_coursework',
        ROOT / 'data/versions/yfinance_review_2026_11_01',
        output / 'offline_quote_audit', require_full=False, refresh=False)
    scope = {'plt': plt, 'index_prices': index_prices}
    exec(helpers['PLOT'], scope)
    metrics = pd.read_csv(folder / 'primary_economic_comparison.csv', index_col=0)
    results = {}
    for name in ['static', 'expanding', 'sliding']:
        eq = pd.read_csv(folder / f'equity_{name}.csv', parse_dates=['Date']).set_index('Date')
        results[name] = dict(equity=eq, metrics=metrics.loc[name].to_dict(),
                             label=name, drawdown=eq.Capital / eq.Capital.cummax() - 1,
                             params={'capital': 100000})
    preds = pd.read_csv(folder / 'predictions/static_rf.csv.gz', parse_dates=['Date'])
    preds = preds.rename(columns={'Random Forest': 'y_prob'})
    config = json.loads((folder / 'experiment_config.json').read_text(encoding='utf-8'))
    result = review_revision.run_backtest(
        preds, quotes, index_prices, horizon=config['label_days'],
        start=config['oos_start'], end=config['oos_end'], top_k=config['top_k'],
        commission=config['commission'], capital=config['capital'],
        long_short=True, label='Static RF long/short')
    # Match the already printed notebook result before replacing its picture.
    assert round(result['metrics']['total_return'] * 100, 2) == -8.49
    assert round(result['metrics']['max_drawdown'] * 100, 2) == -25.50
    results['long_short'] = result
    image_outputs = {}
    for name, result in results.items():
        fig = scope['plot_backtest'](result)
        strategy, benchmark = fig.axes[0].lines[:2]
        assert strategy.get_color() != benchmark.get_color()
        assert benchmark.get_color() == '#e67e22'
        assert fig.axes[1].lines[0].get_color() == '#c0392b'
        image = io.BytesIO()
        fig.savefig(image, format='png', dpi=120, bbox_inches='tight')
        (output / f'backtest_{name}.png').write_bytes(image.getvalue())
        plt.close(fig)
        image_outputs[name] = dict(output_type='display_data', metadata={}, data={
            'image/png': base64.b64encode(image.getvalue()).decode('ascii'),
            'text/plain': [f'Capital and drawdown: {result["label"]}']})
    backup = output / ('backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir()
    for filename in ['sp500_coursework_reviewed.ipynb', 'sp500_coursework_reviewed_colors.ipynb']:
        path = ROOT / 'notebooks' / filename
        shutil.copy2(path, backup / filename)
        nb = json.loads(path.read_text(encoding='utf-8'))
        before = json.loads(path.read_text(encoding='utf-8'))
        nb['cells'][73]['source'] = helpers['replace_function'](
            ''.join(nb['cells'][73]['source'])).splitlines(keepends=True)
        count = 0
        for cell_id, names in [(74, ['static']), (76, ['long_short']),
                                (92, ['expanding', 'sliding'])]:
            outputs = nb['cells'][cell_id]['outputs']
            positions = [i for i, o in enumerate(outputs) if 'image/png' in o.get('data', {})]
            assert len(positions) == len(names), (filename, cell_id, positions)
            for position, name in zip(positions, names):
                outputs[position] = image_outputs[name]
                count += 1
        assert count == 4
        for i, (old, new) in enumerate(zip(before['cells'], nb['cells'])):
            if i != 73:
                assert old['source'] == new['source']
            old_nonimages = [o for o in old.get('outputs', []) if 'image/png' not in o.get('data', {})]
            new_nonimages = [o for o in new.get('outputs', []) if 'image/png' not in o.get('data', {})]
            assert old_nonimages == new_nonimages
        nbformat.validate(nbformat.from_dict(nb))
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(filename, ': replaced 4/4 plots, preserved calculations and text outputs')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, name in zip(axes.flat, results):
        ax.imshow(plt.imread(output / f'backtest_{name}.png'))
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'all_four_plots.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()

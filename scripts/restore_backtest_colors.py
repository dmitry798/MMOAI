"""Update only the backtest drawing function; preserve notebook results."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLOT = '''def plot_backtest(result, color=None):
    eq = result['equity']
    m = result['metrics']
    label = result['label']
    if color is None:
        scheme = label.lower()
        color = ('#9b59b6' if 'long/short' in scheme else
                 '#2ecc71' if 'expanding' in scheme else
                 '#2166ac' if 'sliding' in scheme else '#7b3294')
    market = index_prices.set_index('Date').SP500.reindex(eq.index)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(eq.index, eq.Capital, color=color, linewidth=2,
                 label=label, zorder=3)
    axes[0].plot(eq.index, market / market.iloc[0] * result['params']['capital'],
                 color='#e67e22', linestyle='--', linewidth=1.8,
                 label='S&P 500 (price)')
    axes[0].legend(); axes[0].set_ylabel('Capital')
    axes[0].set_title(label)
    axes[1].plot(eq.index, result['drawdown'], color='#c0392b', linewidth=1)
    axes[1].fill_between(eq.index, result['drawdown'], 0,
                         color='#e74c3c', alpha=.35)
    axes[1].axhline(0, color='#555555', linewidth=.7)
    axes[1].set_ylabel('Drawdown'); axes[1].set_title(f"MDD {m['max_drawdown']:.2%}")
    for ax in axes:
        ax.grid(alpha=.25)
    plt.tight_layout()
    return fig
'''


def replace_function(source):
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'plot_backtest')
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + PLOT + ''.join(lines[node.end_lineno:])


if __name__ == '__main__':
    path = ROOT / 'notebooks/sp500_coursework_reviewed.ipynb'
    nb = json.loads(path.read_text(encoding='utf-8'))
    changed = []
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'def plot_backtest(' in source:
            cell['source'] = replace_function(source).splitlines(keepends=True)
            changed.append(i)
    assert changed == [73], changed
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    builder = ROOT / 'scripts/strengthen_reviewed_notebook.py'
    source = builder.read_text(encoding='utf-8')
    start = source.index('def plot_backtest(')
    end = source.index("\n''')", start)
    builder.write_text(source[:start] + PLOT.rstrip() + source[end:], encoding='utf-8')
    print('Updated cell 73 (UI cell 74) and builder; outputs and calculations preserved.')

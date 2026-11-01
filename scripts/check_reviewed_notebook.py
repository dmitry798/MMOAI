"""Execute all cells on a clearly labelled, small REAL-data smoke configuration.

Does NOT overwrite the reviewed notebook or create article-ready results.
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ['MPLCONFIGDIR']=str(ROOT/'reports/review_smoke/mplconfig')
os.environ['MPLBACKEND']='Agg'
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
nb = nbformat.read(ROOT/'notebooks/sp500_coursework_reviewed.ipynb',as_version=4)
outdir = ROOT/'reports/review_smoke'
outdir.mkdir(parents=True,exist_ok=True)
shell = InteractiveShell.instance()
shell.run_cell("import matplotlib; matplotlib.use('Agg')")

for i,cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    code = cell.source.replace('%matplotlib inline','')
    if i==4:
        code += "\nOOS_START=pd.Timestamp('2023-11-01')\nOOS_END=pd.Timestamp('2024-11-22')\nVIZ_SAMPLE=250\nSHAP_SAMPLE=30\nTOP_K=3\n"
        code += "CACHE_DIR=PROJECT_ROOT/'reports/review_smoke/cache'\nMODEL_DIR=PROJECT_ROOT/'reports/review_smoke/models'\nFIG_DIR=PROJECT_ROOT/'reports/review_smoke/figures'\nPRED_DIR=CACHE_DIR/'predictions'\n"
        code += "for d in [CACHE_DIR,MODEL_DIR,FIG_DIR,PRED_DIR]: d.mkdir(parents=True,exist_ok=True)\n"
        code += "config.update(mode='SMOKE_ONLY',oos_start=str(OOS_START.date()),oos_end=str(OOS_END.date()),top_k=TOP_K,reduced_estimators=True)\n(CACHE_DIR/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')\n"
    if i==9:
        code = """
available = [p for p in YF_DIR.glob('*.csv') if p.stem not in ['download_manifest','INDEX_GSPC']]
chosen = sorted(available)[:12]
assert len(chosen)==12,'Download at least 12 histories before the smoke check'
price_downloaded=pd.concat([pd.read_csv(p,parse_dates=['Date']).assign(Symbol=p.stem) for p in chosen],ignore_index=True)
index_df=pd.read_csv(YF_DIR/'INDEX_GSPC.csv',parse_dates=['Date']).rename(columns={'Close':'SP500'})
index_df=index_df[index_df.Date>='2021-01-01'].copy()
price_downloaded=price_downloaded[price_downloaded.Date>='2021-01-01'].copy()
requested_symbols=sorted(price_downloaded.Symbol.unique())
download_audit=pd.DataFrame({'Symbol':requested_symbols,'status':'smoke_cached_real_data'})
missing_downloads=pd.DataFrame(columns=['Symbol','error'])
factors=rf.prepare_factors(FACTOR_DIR,download=False)
rf.require_coverage(factors,index_df.loc[index_df.Date.between(OOS_START,OOS_END),'Date'])
config.update(requested=len(requested_symbols),loaded=len(requested_symbols),missing=[])
print('SMOKE ONLY: 12 stocks, 2021-2024, reduced estimators, 2000 bootstrap. NOT ARTICLE RESULTS.')
"""
    if i==36:
        code=code.replace('min(5000,','min(300,')
    if i==56:
        code=code.replace('n_estimators=100','n_estimators=5').replace('max_iter=100,','max_iter=5,').replace('time_budget=30','time_budget=1')
    if i==60:
        code=code.replace("'n_estimators': 413","'n_estimators': 5").replace("'n_estimators': 408","'n_estimators': 5")
    if i==95:
        code=code.replace('SHAP_SAMPLE = 2000','SHAP_SAMPLE = 30')
    print(f'RUN CELL {i}',flush=True)
    with capture_output() as captured:
        result=shell.run_cell(code,store_history=True)
    outputs=[]
    if captured.stdout:
        outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:
        outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    cell.outputs=outputs
    cell.execution_count=shell.execution_count-1
    if result.error_in_exec or result.error_before_exec:
        error=result.error_in_exec or result.error_before_exec
        outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=[]))
        nbformat.write(nb,outdir/'smoke_executed.ipynb')
        print(captured.stdout[-5000:],captured.stderr[-2000:])
        raise SystemExit(f'CELL {i} FAILED: {error}')
    nbformat.write(nb,outdir/'smoke_executed.ipynb')
    print(f'PASS CELL {i}',flush=True)
    shell.run_cell('plt.close("all")')
nb.metadata['review_revision']['results']='SMOKE ONLY, 12 stocks, reduced estimators; not article results'
nbformat.write(nb,outdir/'smoke_executed.ipynb')
print('ALL CODE CELLS PASSED — SMOKE ONLY',flush=True)

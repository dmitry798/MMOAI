"""Strengthen the user's reviewed notebook in place, preserving v6/final.

The executed prior notebook is archived before replacing cells. Run this script
only to regenerate the scientific protocol, not to execute model fitting.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path
from textwrap import dedent

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/sp500_coursework_reviewed.ipynb'
backup=ROOT/'reports/review_before_strict_20261101/sp500_coursework_reviewed.ipynb'
if not backup.exists():
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,backup)
nb=json.loads(backup.read_text(encoding='utf-8'))
assert len(nb['cells'])==99
def put(i,text):
    nb['cells'][i]['source']=dedent(text).strip()+'\n'
def source(i):
    return ''.join(nb['cells'][i]['source'])
def change(i,old,new):
    s=source(i)
    if old not in s:
        raise ValueError(f'Cell {i}: missing replacement {old[:60]}')
    put(i,s.replace(old,new))

put(0,'''# Прогнозирование движения акций S&P 500
## Исправленная тетрадь: полный протокол проверки по рецензиям

Я сохраняю 273 признака, исходную 10-дневную цель и набор моделей.
Основной вопрос: меняется ли экономический результат при изменении формы
обучающего окна и правил обновления модели при одинаковых прочих условиях?

**Порядок:** 1. Предметная область → 2. Данные и признаки → 3. Модели,
значимость и экономическое сравнение → 4. SHAP → 5. Итоги и ограничения.
Сохранены 99 ячеек. Предыдущий выполненный прогон сохранён в
`reports/review_before_strict_20261101`; v6 и final не изменены.

Перед долгим обучением выполняются проверки окружения, сохранения файлов,
источников, контрольных сумм и факторных данных. Загрузка при Run All идёт
из зафиксированных локальных файлов, без сетевых запросов.

**Короткие ответы:** k=10, равные доли; 21-дневное удержание использует
тот же 10-дневный сигнал; 260 — прогрев признака, purge train/test равен 10 дням.
Основной результат — риск-корректированная альфа, а не простое превышение индекса.
Новый полный прогон вычисляет все числа заново; результаты старого прогона
не подставляются. Это ретроспективное исследование: прежний OOS уже просмотрен.
''')
put(1,'''---
# 1. Анализ предметной области

## 1.1 Постановка и научный вопрос

Для каждой бумаги я оцениваю вероятность события: хотя бы в одну из следующих
10 торговых сессий её ценовая доходность превысит доходность индекса более чем
на **2 процентных пункта**. Это событие достижения уровня, а не доходность
ровно на десятый день и не гарантия денежной прибыли.

Я проверяю, сохраняются ли различия между стратегиями после выравнивания
периода, модели, расписания обновления, purge, комиссий и правил исполнения.
Это позволяет отделить качество ранжирования от экономического результата.
SHAP помогает описать экспозицию модели к волатильности и экстремумам,
но не доказывает причинность или прибыльность.

## 1.2 Критерии и процедура

- Классификация: ROC-AUC, AP, дневной Precision@10 и его базовая частота.
- Основная экономическая оценка: годовой intercept CAPM с фактическим RF,
  совместный блочный bootstrap разностей; FF5+momentum — проверка экспозиций.
- Дополнительные показатели: Sharpe excess returns, просадка, Profit Factor,
  средний логарифмический рост и накопленная доходность после комиссии.
- OOS фиксирован: 2022-06-02…2024-11-22. CV проводится внутри train:
  пять растущих фолдов по целым датам с purge 10 дней.
- Порог 2 п.п. сохранён из постановки; баланс классов измеряется заново,
  а не считается заранее равным 49/51.

## 1.3 Область применимости

Результат — исследовательское ранжирование и сравнение протоколов. Он не
обещает роста выбранных акций. Фундаментальные данные снимка 2024 не входят
в признаки. Исторический состав индекса не восстановлен, поэтому survivorship
bias остаётся. Проскальзывание, влияние на рынок и дивиденды не моделируются.
''')
put(2,'''---
# 2. Формирование и подготовка данных
## 2.1 Источники

Вселенная — ровно 502 имени из исходного `sp500_companies.csv`.
Основной источник — Yahoo Close (`auto_adjust=False`); подтверждённые
переименования той же бумаги допускаются с сохранением исходного Symbol.
При отсутствии Yahoo используется **целая история** исходного Close или
проверенного архива. Бумага не заменяется компанией-покупателем.
Истории разных источников внутри одного тикера не склеиваются автоматически.

Происхождение каждой истории, даты, SHA256 и доступность будущих цен записаны
в `recovered_download_manifest.csv`. Снимок фиксируется; изменение файлов
после фиксации вызывает ошибку до обучения.

502 запрошенных имени, число загруженных историй и число бумаг с готовыми
признаками — разные величины. IPO нельзя снабжать искусственной предысторией.
Если часть историй не восстановлена, фактическая выборка указывается явно
в каждой итоговой сводке; такой опыт не называется полным опытом на 502.
''')
imports=source(3)
imports=re.sub(r"DATA_VERSION = '[^']+'","DATA_VERSION = 'review_strict_2026_11_01'",imports)
imports=imports.replace("FIG_DIR = PROJECT_ROOT / 'reports' / DATA_VERSION / 'figures'",
                        "FIG_DIR = PROJECT_ROOT / 'reports' / DATA_VERSION / 'figures'")
imports += '''
from src import review_inference as ri
from src import review_protocol as rp
from src import review_factors as rf
from src import review_data as rd
import importlib
# Restart Kernel + Run All is recommended; reload also prevents stale helper code.
for module in (rr,ri,rp,rf,rd):
    importlib.reload(module)
from threadpoolctl import threadpool_limits
environment = rp.preflight(CACHE_DIR)
print('Python kernel:',environment['executable'])
'''
put(3,imports)
put(4,'''# Конфигурация фиксируется до нового обучения; значения не выбираются по P&L.
OOS_START=pd.Timestamp('2022-06-02')
OOS_END=pd.Timestamp('2024-11-22')
LABEL_DAYS=10
HOLD_DAYS,HOLD_DAYS_ROBUST=10,21
INITIAL_CAPITAL,TOP_K,COMMISSION=100_000,10,.001
TRAIN_DAYS,GAP_DAYS,TEST_DAYS,STEP_DAYS=253,10,21,21
BASE_MODEL='Random Forest'
VIZ_SAMPLE,SHAP_SAMPLE=5000,2000
RUN_HYPERPARAM_SEARCH=False
USE_HIST_GB=True
XGB_DEVICE='cpu'
SAVE_PREDICTIONS=True
MIN_MODEL_DAYS=0
# Сохраняю ранее выбранное разрешение явно неполного опыта. Покрытие не скрывается.
# True запрещает обучение до восстановления всех 502 историй.
REQUIRE_FULL_QUOTES=False
REFRESH_DATA=False
N_BOOT=2000
BOOT_BLOCK_DAYS=21  # основной блок; остальные длины — sensitivity, не выбор лучшего p
PRED_DIR=CACHE_DIR/'predictions'
PRED_DIR.mkdir(parents=True,exist_ok=True)
FACTOR_DIR=PROJECT_ROOT/'data/versions/french_factors_review_2026_11_01'
config=dict(protocol='strict_review_v2',mode='retrospective',
    oos_start=str(OOS_START.date()),oos_end=str(OOS_END.date()),label_days=LABEL_DAYS,
    holding=[HOLD_DAYS,HOLD_DAYS_ROBUST],top_k=TOP_K,capital=INITIAL_CAPITAL,
    commission=COMMISSION,train_days=TRAIN_DAYS,gap_days=GAP_DAYS,step_days=STEP_DAYS,
    seed=SEED,bootstrap=N_BOOT,block_days=BOOT_BLOCK_DAYS,model=BASE_MODEL,
    price='Close',auto_adjust=False,execution_lag=1,min_model_days=MIN_MODEL_DAYS,
    require_full_quotes=REQUIRE_FULL_QUOTES,primary_metric='jensen_alpha',
    bootstrap_method=ri.METHOD,parquet_required=False)
assert N_BOOT>=2000
(CACHE_DIR/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(config)
''')
put(9,'''requested_symbols=sorted(companies.Symbol.dropna().unique())
assert len(requested_symbols)==502,'Изменился фиксированный состав исследования'
price_downloaded,index_df,download_audit=rd.prepare_review_data(
    DATA_DIR,YF_DIR,CACHE_DIR,require_full=REQUIRE_FULL_QUOTES,refresh=REFRESH_DATA)
missing_downloads=download_audit.loc[download_audit.status.ne('ok')].copy()
display(download_audit)
if len(missing_downloads):
    print('НЕПОЛНАЯ ВСЕЛЕННАЯ. Не использовать формулировку «эксперимент на 502 акциях».')
    display(missing_downloads[['Symbol','error']])
assert not price_downloaded.duplicated(['Date','Symbol']).any()
factors=rf.prepare_factors(FACTOR_DIR,download=False)
factor_dates=index_df.loc[index_df.Date.between(OOS_START,OOS_END),'Date']
rf.require_coverage(factors,factor_dates)
assert pd.Timestamp(OOS_START) in set(index_df.Date) and pd.Timestamp(OOS_END) in set(index_df.Date)
config.update(requested=len(requested_symbols),loaded=int(price_downloaded.Symbol.nunique()),
              missing=missing_downloads.Symbol.tolist(),factor_source='French daily FF5+Mom+RF')
(CACHE_DIR/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print('Preflight data/factors OK:',config['loaded'],'из',config['requested'])
''')
put(11,'''Историческое окно фиксировано заранее. Сравнение относится только к этому
периоду; оно не гарантирует устойчивости на другом рыночном режиме.''')
put(26,'''### Полнота и пригодность данных

Показываются отдельно: исходная выгрузка, восстановленные истории, готовые
273 признака, наблюдаемые метки и OOS-кандидаты. Наличие котировок после
IPO ещё не означает наличия 260-дневной истории для X_260.
''')
# Keep data/feature formulas and original inferential train restrictions.
cell44=source(44)
cell44 += '''
# Основная проверка связи признаков с классом учитывает зависимость строк:
# ресэмплирую смежные даты вместе со всеми акциями, только внутри train.
feature_block_df=ri.feature_block_tests(eda_train,ALL_FEAT_COLS,
    n_boot=N_BOOT,block=BOOT_BLOCK_DAYS,seed=SEED)
feature_block_df.to_csv(CACHE_DIR/'feature_date_block_tests.csv')
display(feature_block_df.sort_values('q_BY').head(20))
print('Date-block q_BH<.05:',int((feature_block_df.q_BH<.05).sum()))
print('Консервативная q_BY<.05:',int((feature_block_df.q_BY<.05).sum()))
print('ANOVA+BH — описательная диагностика. Основной зависимый анализ — date blocks; '
      'BY учитывает произвольную зависимость между проверками признаков.')
'''
put(44,cell44)
put(43,'''## 2.7 Признаки: ANOVA и проверка зависимости по блокам дат

Прежний «критерий Фишера» — однофакторный ANOVA F-тест, не точный тест
Фишера для таблиц сопряжённости. Для его 273 p-values применяется BH.
Обычные ANOVA-предположения независимости строк в панели нарушены, поэтому
результат описательный и не считается достаточным доказательством сигнала.

Дополнительно я проверяю разность средних признака между классами общим
bootstrap смежных блоков train-дат: внутри дня сохраняются все бумаги.
Публикуются p, BH и более консервативная BY, допускающая зависимость проверок.
Все 273 признака остаются в моделях независимо от этих результатов.
''')
put(46,'''---
# 3. Построение моделей и валидация
## 3.1 Задача и выбор алгоритмов

Задача — бинарная классификация события достижения относительного уровня,
а не прогноз одного ценового ряда. Я сравниваю LR, RF, XGBoost, HistGB и KNN;
FLAML — дополнительный AutoML baseline, soft voting — дополнительный ансамбль.
Это сравнение конкретных конфигураций, а не доказательство превосходства
семейств алгоритмов при оптимальном подборе параметров.
Основная экономическая модель RF зафиксирована заранее.
''')
put(59,'''## 3.4 Дополнительные конфигурации и подбор гиперпараметров

Главный опыт использует одну базовую RF-конфигурацию во всех схемах и SHAP.
LR/RF/XGB с legacy-параметрами показаны дополнительно; они не называются
оптимальными на новой выборке. Новый RandomizedSearchCV запускается только
при явном флаге и использует purged CV внутри train, без выбора по тесту.
HistGB и KNN остаются фиксированными baseline: вычислительный бюджет
ограничен. Такое решение ограничивает выводы сравнением этих конфигураций.
''')
# Replace parquet persistence everywhere with explicit compressed CSV paths.
for i in range(99):
    if nb['cells'][i]['cell_type']!='code':
        continue
    s=source(i)
    s=re.sub(r"expanding_preds\.to_parquet\(PRED_DIR / 'expanding_preds\.parquet', index=False\)",
             "rp.save_predictions(expanding_preds,PRED_DIR/'expanding_preds')",s)
    s=s.replace('expanding_preds.parquet','expanding_preds.csv.gz')
    s=s.replace("sw_preds.to_parquet(PRED_DIR/'sliding_preds.parquet',index=False)",
                "rp.save_predictions(sw_preds,PRED_DIR/'sliding_preds')")
    put(i,s)
# Exclude posthoc sector-picked topK from trading interpretation.
put(68,'''# Дневное ранжирование для диагностики; не глобальный top по всему тесту.
sector_map=dict(zip(companies.Symbol,companies.Sector))
all_preds=expanding_preds[['Symbol','Date','y_true','y_prob']].rename(
    columns={'y_true':'Target','y_prob':'Prob'}).copy()
all_preds=all_preds.merge(df_sorted[['Date','Symbol','X_10']],on=['Date','Symbol'],validate='one_to_one')
all_preds['Sector']=all_preds.Symbol.map(sector_map)
unique_preds=pd.concat([day.nlargest(TOP_K,'Prob') for _,day in all_preds.groupby('Date')],ignore_index=True)
print('Только дневные top-K; диагностика секторов, без торговой фильтрации по Target.')
''')
put(69,'''sector_counts=unique_preds.Sector.value_counts()
sector_accuracy=unique_preds.groupby('Sector').Target.mean()
display(pd.DataFrame({'signals':sector_counts,'class1_rate':sector_accuracy}))
''')
put(71,'''target_date=all_preds.Date.max()
day_preds=all_preds[all_preds.Date.eq(target_date)]
display(day_preds.nlargest(TOP_K,'Prob')[['Symbol','Sector','Prob','Target','X_10']])
''')
put(73,'''TRADING_DAYS_PER_YEAR=252
def run_backtest(*args,**kwargs):
    kwargs.setdefault('top_k',TOP_K)
    kwargs.setdefault('capital',INITIAL_CAPITAL)
    kwargs.setdefault('commission',COMMISSION)
    result=rr.run_backtest(*args,**kwargs)
    result['metrics'].update(rr.factor_risk_metrics(result['equity'],index_prices,factors,lag=BOOT_BLOCK_DAYS))
    rp.validate_backtest(result,kwargs['start'],kwargs['end'])
    return result

def print_backtest(result):
    m=result['metrics']
    print(result['label'],m['start'].date(),m['end'].date())
    print('CAPM alpha/year:',f"{m['jensen_alpha']:+.2%}",'HAC p:',round(m['jensen_p'],4))
    print('FF5+Mom alpha/year:',f"{m['ff_alpha']:+.2%}",'HAC p:',round(m['ff_p'],4))
    print('Beta:',round(m['beta'],3),'IR:',round(m['information_ratio'],3))
    print('Total return:',f"{m['total_return']:+.2%}",
          'Raw gap vs index:',f"{m['alpha_vs_index']:+.2%}")
    print('Sharpe excess:',round(m['sharpe'],3),'MDD:',f"{m['max_drawdown']:.2%}",
          'Profit factor:',round(m['profit_factor'],3),'turnover(two-sided):',round(m['avg_turnover'],3))

def evaluate_strategy(result):
    print('Статистические выводы — только по тестам раздела 3.7; пороги не доказывают готовности к торговле.')

def plot_backtest(result, color=None):
    eq = result['equity']
    m = result['metrics']
    label = result['label']
    if color is None:
        scheme = label.lower()
        color = ('#9b59b6' if 'long/short' in scheme else
                 '#2ecc71' if 'expanding' in scheme else
                 '#3498db' if 'sliding' in scheme else '#34495e')
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
''')
put(75,'''Оценка стратегии включает риск-корректированную альфу, excess-return
Sharpe, ежедневную просадку и Profit Factor по реализованным денежным P&L.
Заранее установленные «хорошие» пороги не заменяют статистическую проверку.
CAPM использует дневной RF; факторная модель служит проверкой экспозиций.
''')
put(76,'''# Long/short — дополнительная диагностика, не главный результат статьи.
res_long_short=run_backtest(bt_preds,panel,index_prices,horizon=HOLD_DAYS,
    start=OOS_START,end=OOS_END,long_short=True,label='Static RF long/short')
print_backtest(res_long_short)
plot_backtest(res_long_short); plt.show()
''')
put(88,'''## 3.6 Статистические различия между алгоритмами

Использую общие circular moving-block выборки дат (основной блок 21 день,
не менее 2 000 повторов). Переношу весь день со всеми акциями и предсказаниями.
Оценка — pooled ROC-AUC; пофолдовое среднее — отдельная величина.

CI и p получены инверсией одного basic equal-tail теста центрированной
bootstrap-ошибки, с plus-one и проверкой конечновыборочной согласованности.
Публикуются номинальные 95% интервалы и отдельно Holm-adjusted p; номинальный
интервал не является одновременным интервалом после поправки.

Для 10 пар пяти заданных моделей применяется Holm. Значимость заявляется
по согласованному правилу bootstrap + приближённый NB-тест по фолдам.
Уилкоксон — чувствительность; DeLong — только iid-диагностика, не точный
тест для зависимой панели и не гарантированная нижняя граница ошибки.
Отдельная проверка AUC против .5 имеет своё семейство из пяти гипотез.
''')
# Retain DeLong and NB interfaces; replace all old bootstrap implementations.
s=source(89)
mid=s[s.index('def _midrank'):s.index('def block_bootstrap_aucs')]
rest=s[s.index('def corrected_paired_ttest'):]
rest=rest.replace("w_stat, w_p = wilcoxon(sa, sb, zero_method='wilcox',\n                                   alternative='two-sided')",
    "mask=np.isfinite(sa)&np.isfinite(sb)\n            sa,sb=sa[mask],sb[mask]\n            w_p=wilcoxon(sa,sb,zero_method='wilcox',alternative='two-sided').pvalue if len(sa)>=3 and np.any(sa!=sb) else 1.")
rest=rest.replace("p = 2 * (1 - student_t.cdf(abs(t_stat), df=k - 1))","p = float(2*student_t.sf(abs(t_stat),df=k-1))")
rest=rest.replace("t_stat = mean_d / se if se > 0 else 0.0","t_stat = mean_d/se if se>0 else (0. if mean_d==0 else np.inf)")
# Holm must preserve failed/non-estimable tests; do not use stale inline version.
start=rest.index('def holm_correction')
end=rest.index('def significance_report')
rest=rest[:start]+"def holm_correction(pvalues):\n    return ri.holm(pvalues)\n\n\n"+rest[end:]
put(89,mid+'''
def block_bootstrap_aucs(df,model_names,n_boot=2000,alpha=.05,seed=SEED):
    return ri.block_bootstrap_aucs(df,model_names,n_boot=n_boot,block=BOOT_BLOCK_DAYS,alpha=alpha,seed=seed)
def bootstrap_diff(boot,name_a,name_b):
    return ri.bootstrap_diff(boot,name_a,name_b)

'''+rest)
put(90,'''model_names=list(models_sw)
scored_sw_preds=sw_preds.dropna(subset=['y_true']).copy()
with threadpool_limits(limits=1):
    boot=block_bootstrap_aucs(scored_sw_preds,model_names,n_boot=N_BOOT)
auc_signal_df=pd.DataFrame([{'Модель':name,'AUC':boot['observed'][name],
    **boot['inference'][name]} for name in model_names])
auc_signal_df['p_Holm']=ri.holm(auc_signal_df.p)
display(auc_signal_df)
auc_signal_df.to_csv(CACHE_DIR/'auc_vs_random_tests.csv',index=False)
sig_df=significance_report(sw_fold_scores,
    pooled_preds=scored_sw_preds[['Date','y_true']+model_names],
    n_train=TRAIN_DAYS,n_test=sw_preds.groupby('fold').Date.nunique().mean(),boot=boot)
sig_df['Значимо']=(sig_df['Bootstrap p (Holm)']<.05)&(sig_df['NB p (Holm)']<.05)
display(sig_df)
sig_df.to_csv(CACHE_DIR/'significance_tests.csv',index=False)
print('Значимых пар по двум критериям:',int(sig_df['Значимо'].sum()),'из',len(sig_df))
print('Интервалы номинальные; решения после множественности — по Holm p, '
      'а не по одному только пересечению 95% ДИ с нулём.')
''')
put(91,'''## 3.7 Экономическое сравнение и статистическая значимость

Я разделяю три сравнения: форму окна (expanding/sliding при обновлении каждые
21 день), обновление доступной истории (expanding/static) и удержание 10/21
день на общей дневной сетке перекрывающихся долей. Модель и 10-дневная метка
одинаковы. Разный объём истории — часть определения формы окна.

Портфель: k=10, равные доли. Top-k фиксируется после закрытия t, вход — по
закрытию t+1; недоступная цена входа оставляет долю cash, без продвижения
следующей бумаги по факту будущей доступности. Выход через h сессий, только
внутри общего OOS. Переоценка ежедневная; первая потеря входит в просадку.
Позиции ликвидируются полностью, даже повторная покупка платит комиссию.
Оборот обычной стратегии: (покупки+продажи)/капитал перед циклом, two-sided.
Для overlapping дополнительно указывается дневной оборот совокупных долей.

**Первичный показатель:** годовая CAPM alpha с реальным RF; FF5+Mom —
дополнительный контроль экспозиций. Sharpe считается по excess returns.
Простое превышение доходности индекса показано отдельно; это не Jensen alpha.
Используются ценовые доходности без дивидендов; премии French являются
факторными total-return рядами, поэтому FF-регрессия — проверка экспозиций,
а не заявление о реализованной total-return альфе.

Парные bootstrap-тесты пересчитывают регрессии и все показатели на общих
блоках дат. Проверяются CAPM/FF alpha, annual log growth, Sharpe, MDD,
Profit Factor и накопленная доходность. PF — сумма положительных денежных
P&L / сумма абсолютных отрицательных P&L; bootstrap переносит совместные
дневные суммы P&L на датах закрытия. Этот интервал условен на наблюдаемом
пути номиналов и не имитирует заново всю стратегию после перемешивания.
Модель также не переобучается внутри bootstrap: вывод условен на прогнозах.

Основной блок — 21 день; длины 2/3/5/8 периодов проверяются отдельно как
чувствительность. Семейства Holm объявлены по типу сравнения: primary alpha
отдельно, остальные показатели — вторичное семейство. Индивидуальные тесты
Sharpe/alpha против нуля и PF против единицы имеют отдельное семейство.
MDD против нуля — только диагностика наличия потерь, не тест преимущества.

Накопленная доходность и MDD — вторичные оценки фиксированного окна;
annual log growth удобнее для сравнения процесса доходностей. Никакой
bootstrap не гарантирует точности при произвольной нестационарности.
''')
s=source(92)
# Existing predictions/fit protocol is preserved, with new artifact directory.
s=s.replace("runs,robust_runs,cost_rows = {},{},[]","runs,robust_runs,cost_rows,cost_runs = {},{},[],{}")
s=s.replace("cost_rows.append({'scheme':scheme,'holding':h,'cost_multiplier':multiplier,**result['metrics']})",
            "cost_runs[(scheme,h,multiplier)]=result\n            cost_rows.append({'scheme':scheme,'holding':h,'cost_multiplier':multiplier,**result['metrics']})")
s=s.replace("main_cols = ['jensen_alpha','jensen_p','beta','information_ratio','total_return',",
            "main_cols = ['jensen_alpha','jensen_p','ff_alpha','ff_p','beta','information_ratio','total_return',")
s += '''
for key,result in cost_runs.items():
    rp.validate_backtest(result,common_start,common_end)
    if len(result['entry_audit']):
        result['entry_audit'].to_csv(CACHE_DIR/f'entry_audit_{key[0]}_h{key[1]}_cost{key[2]}.csv',index=False)
rp.save_predictions(exp_monthly,PRED_DIR/'expanding_retrained')
rp.save_predictions(static_preds,PRED_DIR/'static_rf')
'''
put(92,s)
put(93,'''# Matched CI/p, common daily dates, actual RF, refitted factor regressions.
def annotated_tests(table,category,comparison,h,block):
    table=table.copy()
    table.attrs={}
    table['category']=category
    table['comparison']=comparison
    table['holding']=h
    table['primary_block']=block==BOOT_BLOCK_DAYS
    table['family']=category+np.where(table.metric.eq('jensen_alpha'),':primary_CAPM',':secondary_metrics')
    rp.validate_inference(table)
    return table

def adjust_families(table):
    table=table.copy()
    table['p_Holm']=np.nan
    # Sensitivity blocks are not searched for a winning p-value.
    for _,positions in table.groupby(['family','block']).groups.items():
        table.loc[positions,'p_Holm']=ri.holm(table.loc[positions,'p'])
    table['significant_Holm']=table.p_Holm.lt(.05)&table.status.eq('ok')
    rp.validate_inference(table)
    return table

scheme_rows,individual_rows,cost_rows_tests=[],[],[]
with threadpool_limits(limits=1):
    for h,group in [(HOLD_DAYS,runs),(HOLD_DAYS_ROBUST,robust_runs)]:
        blocks=sorted(set([BOOT_BLOCK_DAYS]+[h*k for k in [2,3,5,8]]))
        for block in blocks:
            for a,b in [('expanding','sliding'),('expanding','static'),('sliding','static')]:
                table=ri.economic_bootstrap(group[a],group[b],index_prices,factors=factors,
                    n_boot=N_BOOT,block=block,seed=SEED)
                scheme_rows.append(annotated_tests(table,'schemes',a+' - '+b,h,block))
            # Individual evidence, not a profit claim from a Sharpe point estimate.
            if block==BOOT_BLOCK_DAYS:
                for scheme,result in group.items():
                    table=ri.individual_bootstrap(result,index_prices,factors=factors,
                        n_boot=N_BOOT,block=block,seed=SEED)
                    individual_rows.append(annotated_tests(table,'individual',scheme,h,block))
        for scheme in group:
            for multiplier in [3,5]:
                stressed=cost_runs[(scheme,h,multiplier)]
                base=cost_runs[(scheme,h,1)]
                table=ri.economic_bootstrap(stressed,base,index_prices,factors=factors,
                    n_boot=N_BOOT,block=BOOT_BLOCK_DAYS,seed=SEED)
                cost_rows_tests.append(annotated_tests(table,'cost_change',f'{scheme}: {multiplier}x - 1x',h,BOOT_BLOCK_DAYS))
                table=ri.individual_bootstrap(stressed,index_prices,factors=factors,
                    n_boot=N_BOOT,block=BOOT_BLOCK_DAYS,seed=SEED)
                individual_rows.append(annotated_tests(table,'individual_stress',f'{scheme} cost{multiplier}',h,BOOT_BLOCK_DAYS))

alpha_tests=adjust_families(pd.concat(scheme_rows,ignore_index=True))
individual_tests=adjust_families(pd.concat(individual_rows,ignore_index=True))
cost_tests=adjust_families(pd.concat(cost_rows_tests,ignore_index=True))
alpha_tests.to_csv(CACHE_DIR/'scheme_all_metrics_significance.csv',index=False)
individual_tests.to_csv(CACHE_DIR/'individual_metrics_significance.csv',index=False)
cost_tests.to_csv(CACHE_DIR/'commission_significance.csv',index=False)
print('Основные сравнения схем: блок21, CAPM alpha')
display(alpha_tests[alpha_tests.primary_block&alpha_tests.metric.eq('jensen_alpha')])
print('Вторичные показатели схем, блок21')
display(alpha_tests[alpha_tests.primary_block&~alpha_tests.metric.eq('jensen_alpha')])
print('Индивидуальные alpha/Sharpe/PF, блок21')
display(individual_tests[individual_tests.metric.isin(['jensen_alpha','ff_alpha','sharpe','profit_factor'])])
print('Комиссии: изменение primary alpha')
display(cost_tests[cost_tests.metric.eq('jensen_alpha')])

# Overlapping 1/h sleeves: same daily calendar for both holding horizons.
overlap_results,overlap_rows={},[]
with threadpool_limits(limits=1):
    for scheme,p in pred_sources.items():
        pair={}
        for h in [HOLD_DAYS,HOLD_DAYS_ROBUST]:
            result=rr.overlapping_backtest(prep(p),panel,index_prices,horizon=h,
                start=common_start,end=common_end,top_k=TOP_K,commission=COMMISSION,
                capital=INITIAL_CAPITAL,label=scheme+' overlapping')
            result['metrics'].update(rr.factor_risk_metrics(result['equity'],index_prices,factors,lag=BOOT_BLOCK_DAYS))
            rp.validate_backtest(result,common_start,common_end)
            pair[h]=result
            result['equity'].to_csv(CACHE_DIR/f'overlap_equity_{scheme}_h{h}.csv')
            result['trades'].to_csv(CACHE_DIR/f'overlap_trades_{scheme}_h{h}.csv',index=False)
        overlap_results[scheme]=pair
        for block in sorted(set([BOOT_BLOCK_DAYS]+[HOLD_DAYS_ROBUST*k for k in [2,3,5,8]])):
            table=ri.economic_bootstrap(pair[HOLD_DAYS_ROBUST],pair[HOLD_DAYS],index_prices,
                factors=factors,n_boot=N_BOOT,block=block,seed=SEED)
            overlap_rows.append(annotated_tests(table,'holding_overlap',scheme+' h21 - h10',21,block))
overlap_tests=adjust_families(pd.concat(overlap_rows,ignore_index=True))
overlap_tests.to_csv(CACHE_DIR/'overlapping_horizon_all_metrics.csv',index=False)
display(overlap_tests[overlap_tests.primary_block])
print('Отсутствие значимости не доказывает эквивалентности. '
      'Номинальный CI сравнивается с raw p; вывод после множественности — по p_Holm.')
''')
shap=source(95)
shap=shap[shap.index('from matplotlib.patches import Patch'):]
shap=shap.replace('SHAP_SAMPLE = 2000','SHAP_SAMPLE = min(SHAP_SAMPLE,len(X_test))')
put(95,'# Та же базовая RF-конфигурация, что в основном экономическом опыте.\n'+shap)
put(96,'''SHAP описывает базовый статический RF той же конфигурации, что применяется
во всех экономических схемах. Интерпретируется один fit и фиксированная
подвыборка; устойчивость направлений эффекта между всеми фолдами не заявляется.
Ранг |SHAP| не определяет знак; знак читается по текущему beeswarm.
SHAP помогает понять факторную экспозицию, но не доказывает превосходство окна.
''')
put(97,'''# Все проверки выполняются до отметки completed.
for name,p in pred_sources.items():
    rp.validate_predictions(p)
for table in [alpha_tests,individual_tests,cost_tests,overlap_tests]:
    rp.validate_inference(table)
assert train_df.LabelEnd.max()<test_df.Date.min()
assert len(ALL_FEAT_COLS)==273 and ALL_FEAT_COLS==rr.ALL_FEATURES
assert config['loaded']==price_downloaded.Symbol.nunique()
assert all(r['equity'].index.equals(runs['expanding']['equity'].index) for r in runs.values())
print('СВОДКА НОВОГО ПРОГОНА')
print('Запрошено:',len(requested_symbols),'историй:',price_downloaded.Symbol.nunique(),
      'бумаг с полным Target:',daily_df.Symbol.nunique(),'строк:',len(daily_df))
print('Класс0:',1-daily_df.Target.mean(),'класс1:',daily_df.Target.mean())
display(funnel_df)
display(cv_results_df.drop(columns='folds'))
display(sw_results_df)
display(comparison_df[main_cols])
display(alpha_tests[alpha_tests.primary_block&alpha_tests.metric.eq('jensen_alpha')])
display(overlap_tests[overlap_tests.primary_block&overlap_tests.metric.eq('jensen_alpha')])
print('Feature date-block BH:',int(feature_block_df.q_BH.lt(.05).sum()),
      'BY:',int(feature_block_df.q_BY.lt(.05).sum()))
print('Согласованных значимых пар алгоритмов:',int(sig_df['Значимо'].sum()))
config['eligible']=int(daily_df.Symbol.nunique())
config['rows']=len(daily_df)
config['environment']=environment
config['all_502_histories']=bool(price_downloaded.Symbol.nunique()==len(requested_symbols))
config['inference_integrity']='matched_CI_p_all_passed'
config['limitations']=['retrospective_OOS','conditional_on_fitted_predictions',
    'survivorship_bias','price_returns_without_dividends','PF_conditional_observed_notional_path',
    'block_stationarity_approximation']
(CACHE_DIR/'run_completed.json').write_text(json.dumps({'status':'completed','config':config},indent=2),encoding='utf-8')
print('Расчёт завершён. Источник чисел для статьи — CSV этого каталога:',CACHE_DIR)
''')
put(98,'''# 5. Итоги и ограничения

Я формулирую численные выводы только по предыдущей сводке и сохранённым CSV.
Главный вывод о форме окна относится к expanding/sliding с одинаковым
расписанием, purge, моделью и OOS. Основная проверка — CAPM alpha;
FF5+momentum, Sharpe, просадка, PF и log growth дополняют её.

1. Положительная доходность или Precision@10 выше baseline не доказывают
   риск-корректированного преимущества. Достижение цели не равно P&L сделки.
2. После множественных сравнений вывод делается по заранее определённому
   Holm-семейству. Номинальный 95% CI не выдаётся за одновременный интервал.
3. 21-дневное удержание — исполнение исходного 10-дневного сигнала, а не
   сравнение моделей с разными метками. Overlapping учитывает cash на краях.
4. Фактическая полнота и пригодность данных указаны отдельно. Неполная
   вселенная не становится полной от увеличения числа строк.
5. Исторический состав индекса не восстановлен; survivorship bias остаётся.
   Пропущенные/делистингованные истории не заменяются историями покупателей.
6. Основной RF и SHAP имеют одну конфигурацию. Дополнительные legacy tuning
   параметры не объявляются оптимальными на новой вселенной.
7. Доходности ценовые, без дивидендов; нет slippage/impact/borrow fees.
   FF-факторы — total-return премии: регрессия контролирует экспозиции, а не
   устраняет все различия определения доходности.
8. Bootstrap условен на прогнозах и наблюдаемом пути P&L, не повторяет fit
   модели и выбор экспериментов. Длины блоков — чувствительность; при
   нестационарности и небольшом числе независимых блоков мощность ограничена.
9. OOS уже использовался в предыдущих версиях. Это ретроспективная оценка;
   строгое независимое подтверждение потребует нового периода/выборки.

Публикация кода подготовлена локально, push не выполняется. Для статьи
остаются редакторские изменения: ссылки на таблицы, раздельные таблицы
scheme/horizon/cost, перенос собственных результатов из Related Work,
обоснование или замена нерелевантных источников. PDF в этой задаче не изменяется.
''')

# Remove stale quantitative/runtime claims, including a CV/tuned-test mismatch.
for i in [56,60]:
    s=source(i)
    s=s.replace('~9 с против ~11 мин на одну ', 'реализация с биннингом; время одной ')
    s=s.replace('подгонку при 273 признаках,','подгонки зависит от окружения,')
    s=s.replace('на 225 тыс. наблюдений и','на обучающей выборке с')
    put(i,s)
s=source(61)
s=s.replace("cv_val = cv_lookup.get(tuned_to_base.get(name, name), float('nan'))",
    "cv_val = cv_lookup.get(BASE_MODEL,float('nan')) if name=='Random Forest (base)' else float('nan')")
s=s.replace("flag = '  ! drift' if abs(gap) > 0.03 else ''", "flag = ''")
s=s.replace("print('\\nGap > 0.03 (!) указывает на distribution shift между обучением и тестом.')",
    "print('\\nCV/test gap показан только для одинаковой базовой RF-конфигурации; это описание, не тест drift.')")
s=s.replace('Предсказания expanding сохранены','Статические прогнозы сохранены')
put(61,s)
put(50,source(50).replace('225 тыс. точек','Большое число точек'))
put(85,source(85).replace('при 94 фолдах маркеры сливаются','при большом числе фолдов маркеры сливаются'))
put(38,source(38).replace("print('Соседние горизонты (X_1↔X_5, X_5↔X_10) сильно коррелируют.')",
                       "print('Корреляции описывают текущий train и не доказывают предсказуемость.')")
                  .replace("print('Далёкие (X_1-X_260) - слабо')","print('Выводы о группах признаков делаются по текущей матрице.')"))

# Scrub stale output/state, preserve cells and IDs, add protocol metadata.
for i,cell in enumerate(nb['cells']):
    cell['source']=source(i).splitlines(keepends=True)
    cell.setdefault('id',f'review-{i:03d}')
    if cell['cell_type']=='code':
        cell['outputs']=[]
        cell['execution_count']=None
    cell['metadata']={}
nb['metadata']['review_revision']['protocol']='strict_review_v2'
nb['metadata']['review_revision']['results']='unexecuted; prior executed notebook archived'
nb['metadata']['review_revision']['backup']=str(backup.relative_to(ROOT))
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Strengthened',path.name,'99 cells')

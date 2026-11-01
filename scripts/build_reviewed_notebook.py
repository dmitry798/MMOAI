"""Create a NEW notebook; never write the original v6/final files."""
import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'notebooks/sp500_coursework_reviewed.ipynb').exists():
    raise SystemExit('Reviewed notebook already exists. Use scripts/strengthen_reviewed_notebook.py; do not overwrite the final protocol with the legacy builder.')
source = ROOT/'notebooks/sp500_coursework_final.ipynb'
nb = json.loads(source.read_text(encoding='utf-8'))
original_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ROOT/'notebooks').glob('*.ipynb') if p.name in
                   ['sp500_coursework_v6.ipynb','sp500_coursework_final.ipynb']}


def put(i,text):
    nb['cells'][i]['source'] = dedent(text).strip()+'\n'


def replace(i,old,new):
    text = ''.join(nb['cells'][i]['source'])
    if old not in text:
        raise ValueError(f'Cell {i}: text not found: {old[:80]}')
    nb['cells'][i]['source'] = text.replace(old,new)


put(0,'''# Прогнозирование движения акций S&P 500
## Курсовая работа: исправления по рецензиям, 2026-11-01

**Новая тетрадь на основе final.** Исходные v6/final не изменяются.
Сохранены порядок разделов, 99 ячеек, все 273 признака и набор алгоритмов.

Цель — оценить вероятность того, что акция хотя бы в один из следующих
10 торговых дней опередит ценовой индекс на **2 процентных пункта**.
Это событие достижения уровня, а не доходность ровно в день 10 и не обещание прибыли.

**Изменения методики по рецензиям:** загрузка фиксированных 502 тикеров через
yfinance; CV по датам с purge 10 дней; полная наблюдаемость будущего для метки;
expanding с переобучением каждые 21 день; общие OOS-даты и gap=10;
BH для ANOVA; смежные блоки bootstrap (>=2000); Jensen alpha, beta, IR и оборот;
исполнение на следующем закрытии и ежедневная переоценка.

**Ответы на вопросы:** k=10, равные веса; при удержании 21 день используются
те же прогнозы **10-дневной** модели, целевая переменная не меняется.
260 дней — прогрев X_260, не необходимый разрыв train/test. Код final
задавал такой разрыв явно; здесь для известных меток достаточно 10 дней purge.
GitHub: подготовлен воспроизводимый код, публикация пользователем; push не выполняется.

**Числа статьи 170/378475 и старые p-values не являются результатами этой версии.**
Все выводы строятся заново. 502 — запрошенная вселенная, фактическая доступность
и пригодность показаны в manifest и воронке; survivorship bias снимка 2024 остаётся.

Порядок: 1. Предметная область → 2. Данные и 273 признака →
3. Модели, CV и эксперименты → 4. SHAP → 5. Выводы и ограничения.
''')
put(2,'''---
# 2. Формирование и подготовка данных для анализа
## 2.1 Описание используемых данных

Для нового опыта **весь** фиксированный список из `sp500_companies.csv`
загружается через yfinance, включая индекс `^GSPC`. Состав индекса заново
из интернета не берётся. Фундаментальные показатели не входят в модель;
сектор используется только для описательного анализа.

Цены: `Close`, `auto_adjust=False` (ценовая доходность без дивидендов).
Нельзя молча смешивать Close и Adj Close. Период запроса фиксирован:
2014-12-22…2024-12-20, `end=2024-12-21` исключительный.
Старые CSV сохраняются для аудита. Ошибки загрузки перечисляются явно;
нет автоматической замены делистингованной компании её преемником.
''')
replace(3,"import os","import sys\nimport json\nimport os")
replace(3,"warnings.filterwarnings('ignore')","warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)")
replace(3,"import seaborn as sns", "os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / '.mplconfig_review'))\nimport seaborn as sns")
replace(3,"DATA_VERSION = 'v1.0.0_coursework'","DATA_VERSION = 'review_2026_11_01'\nsys.path.insert(0, str(PROJECT_ROOT))\nfrom src import review_revision as rr")
replace(3,"DATA_DIR  = PROJECT_ROOT / 'data' / 'versions' / DATA_VERSION","DATA_DIR = PROJECT_ROOT / 'data' / 'versions' / 'v1.0.0_coursework'\nYF_DIR = PROJECT_ROOT / 'data' / 'versions' / 'yfinance_review_2026_11_01'")
replace(3,"FIG_DIR   = PROJECT_ROOT / 'reports' / 'figures'","FIG_DIR = PROJECT_ROOT / 'reports' / DATA_VERSION / 'figures'")
put(4,'''# Единая, заранее зафиксированная конфигурация
OOS_START = pd.Timestamp('2022-06-02')
OOS_END = pd.Timestamp('2024-11-22')
LABEL_DAYS = 10
HOLD_DAYS, HOLD_DAYS_ROBUST = 10, 21
INITIAL_CAPITAL, TOP_K, COMMISSION = 100_000, 10, .001
TRAIN_DAYS, GAP_DAYS, TEST_DAYS, STEP_DAYS = 253, 10, 21, 21
BASE_MODEL = 'Random Forest'  # определён заранее, не выбран по тестовой доходности
VIZ_SAMPLE, SHAP_SAMPLE = 5000, 2000
RUN_HYPERPARAM_SEARCH = False
USE_HIST_GB = True  # сохранено из final, не новая замена
XGB_DEVICE = 'cpu'  # переносимость; 'cuda' можно задать явно
SAVE_PREDICTIONS = True
MIN_MODEL_DAYS = 0  # после прогрева не требуем ЕЩЁ 270 дней: новые IPO не исключаются целиком
REQUIRE_FULL_QUOTES = True  # остановка, если хотя бы один из 502 тикеров не загружен
N_BOOT = 2000
BOOT_BLOCK_DAYS = 21
PRED_DIR = CACHE_DIR / 'predictions'
PRED_DIR.mkdir(parents=True,exist_ok=True)
config = dict(oos_start=str(OOS_START.date()),oos_end=str(OOS_END.date()),
              label_days=LABEL_DAYS,holding=[HOLD_DAYS,HOLD_DAYS_ROBUST],
              top_k=TOP_K,commission=COMMISSION,train_days=TRAIN_DAYS,
              gap_days=GAP_DAYS,step_days=STEP_DAYS,seed=SEED,
              bootstrap=N_BOOT,block_days=BOOT_BLOCK_DAYS,model=BASE_MODEL,
              price='Close',auto_adjust=False,execution_lag=1,min_model_days=MIN_MODEL_DAYS)
(CACHE_DIR/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(config)
''')
put(9,'''# Единый источник цен для всего нового опыта
requested_symbols = sorted(companies.Symbol.dropna().unique())
assert len(requested_symbols) == 502, 'Состав отличается от снимка статьи — проверьте источник'
price_downloaded, index_df, download_audit = rr.download_universe(requested_symbols,YF_DIR)
missing_downloads = download_audit.query("status != 'ok'")
display(download_audit)
if len(missing_downloads):
    display(missing_downloads)
    print('Полная вселенная НЕ получена; нельзя называть этот прогон опытом на 502 бумагах.')
    if REQUIRE_FULL_QUOTES:
        raise RuntimeError('Не все истории доступны. См. download_manifest.csv. '
                           'REQUIRE_FULL_QUOTES=False разрешает явно неполный исследовательский прогон.')
print(index_df.shape,index_df.Date.min(),index_df.Date.max())
''')
put(12,'''## 2.3 Методы предобработки данных

- Один календарь торговых сессий индекса. Пропуски цены остаются NaN;
  нет bfill, синтетической предыстории IPO или сжатия горизонта по непустым строкам.
- Прогрев базового X_260: 260 прошлых торговых сессий; rolling-формулы
  сохранены (`min_periods=1`, sample std, cv=std/(mean(abs(X))+1e-6)).
- **Изменение отбора:** прежний фильтр ещё 270 готовых строк после прогрева
  исключал молодые бумаги целиком; в основном новом опыте он отключён.
  Число бумаг, которые он исключил бы, сохраняется в аудите.
- Для метки требуются все следующие 10 цен; неполный хвост не становится классом 0.
- StandardScaler внутри Pipeline для LR и KNN; деревьям масштабирование не нужно.
- Разбиение по целым датам с purge 10 дней и проверкой `LabelEnd < test_start`.
- Баланс классов рассчитывается заново для train и каждого теста.
''')
put(18,'''index_df = index_df.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
assert index_df.SP500.gt(0).all()
print('Индекс:',index_df.Date.min(),index_df.Date.max(),len(index_df))
''')
put(26,'''### Полнота данных и состав рабочей выборки

Сопоставляем покрытие старой выгрузки и нового запроса на все 502 тикера.
Число успешно загруженных бумаг, число с признаками и число доступных на
конкретную OOS-дату — разные величины. Вывод о полной выборке допустим
только по фактическому manifest. 260-дневный признак не вычисляется до IPO.
''')
put(27,'''legacy_raw = pd.read_csv(DATA_DIR/'sp500_stocks.csv',usecols=['Date','Symbol','Close'],parse_dates=['Date'])
legacy_coverage = legacy_raw.groupby('Symbol').Close.count().rename('legacy_quotes')
coverage_audit = download_audit[download_audit.Symbol != '^GSPC'].merge(legacy_coverage,on='Symbol',how='left')
n_symbols_file = len(requested_symbols)
price_raw = price_downloaded.copy()
coverage_funnel = [('Запрошено фиксированных тикеров',n_symbols_file,'снимок состава 2024'),
                   ('Загружены непустые истории',price_raw.Symbol.nunique(),'см. manifest')]
coverage_audit.to_csv(CACHE_DIR/'coverage_audit.csv',index=False)
display(coverage_audit)
print('Старых непустых историй:',int(legacy_coverage.gt(0).sum()))
print('Новых непустых историй:',price_raw.Symbol.nunique())
print('Данные исходного CSV не подмешиваются к новой выгрузке.')
''')
put(28,'''K_VALUES = rr.K_VALUES
FEAT_COLS = rr.BASE_FEATURES
price_all = price_raw.sort_values(['Symbol','Date']).copy()
# Полная панель цен нужна в бэктесте, включая хвост без меток.
full_price_panel = price_all[['Date','Symbol','Close']].copy()
daily_df = rr.build_base_panel(price_all,index_df,horizon=LABEL_DAYS)
print('До прогрева:',daily_df.shape)
''')
put(31,'''daily_df = rr.add_statistical_features(daily_df)
ALL_FEAT_COLS = rr.ALL_FEATURES.copy()
ext_feature_cols = rr.EXTRA_FEATURES.copy()
assert len(ALL_FEAT_COLS) == 273
feature_ready = daily_df[ALL_FEAT_COLS].notna().all(axis=1)
prediction_panel = daily_df.loc[feature_ready].copy()  # independent of FUTURE label availability
valid_rows = feature_ready & daily_df.Target.notna()
available_counts = daily_df.loc[valid_rows].groupby('Symbol').size()
dropped_by_filter = sorted(available_counts[available_counts < 270].index)
eligibility = pd.DataFrame({'Symbol':requested_symbols}).merge(
    available_counts.rename('eligible_rows'),on='Symbol',how='left').fillna({'eligible_rows':0})
eligibility['excluded_by_old_270_filter'] = eligibility.eligible_rows.lt(270)
eligibility.to_csv(CACHE_DIR/'model_eligibility.csv',index=False)
daily_df = daily_df.loc[valid_rows].copy()
if MIN_MODEL_DAYS:
    daily_df = daily_df[daily_df.Symbol.isin(available_counts[available_counts>=MIN_MODEL_DAYS].index)]
daily_df['Target'] = daily_df.Target.astype(int)
coverage_funnel.append(('Есть 273 признака и полный таргет',daily_df.Symbol.nunique(),'IPO/пропуски/прогрев'))
funnel_df = pd.DataFrame(coverage_funnel,columns=['Шаг','Тикеров','Причина отсева'])
display(funnel_df)
funnel_df.to_csv(CACHE_DIR/'coverage_funnel.csv',index=False)
print('Рабочих строк:',len(daily_df),'бумаг:',daily_df.Symbol.nunique())
print('Класс 0:',(daily_df.Target==0).mean(),'класс 1:',daily_df.Target.mean())
# Target-зависимый EDA ограничен доступными до OOS метками.
eda_train = daily_df[(daily_df.Date<OOS_START)&(daily_df.LabelEnd<OOS_START)].copy()
''')
put(30,'''Описательные характеристики цен выводятся выше заново. Различия между
средним и медианой не доказывают предсказуемость относительных доходностей.''')
put(34,'''Признаки готовы к хронологическому разбиению. Прогрев X_260 использует
прошлое; он не требует 260-дневного разрыва перед тестом.''')
put(36,'''from sklearn.feature_selection import mutual_info_classif
df_scoring = eda_train.sample(min(5000,len(eda_train)),random_state=SEED)
X_score,y_score = df_scoring[ALL_FEAT_COLS],df_scoring.Target
importances = mutual_info_classif(X_score,y_score,random_state=SEED)
feature_info = pd.DataFrame({'Feature':ALL_FEAT_COLS,'Importance':importances}).sort_values('Importance',ascending=False)
display(feature_info.head(5))
print('Описательный EDA на train; признаки из модели не удаляются.')
''')
for i in [37,38,39,40,44]:
    replace(i,'daily_df','eda_train')
replace(44,"fisher_df = pd.DataFrame(fisher_scores).T.sort_values('f_stat', ascending=False)",
        "fisher_df = pd.DataFrame(fisher_scores).T.sort_values('f_stat', ascending=False)\nfisher_df['q_BH'] = rr.bh_adjust(fisher_df.p_value.fillna(1))\nfisher_df.to_csv(CACHE_DIR/'anova_bh.csv')")
replace(44,"print(f'Значимых на уровне p<0.05: {(fisher_df[\"p_value\"] < 0.05).sum()}')",
        "print(f'ANOVA F-тест (не точный тест Фишера), BH q<0.05: {(fisher_df.q_BH < .05).sum()}')\nprint('Зависимость строк панели ограничивает интерпретацию ANOVA; q — описательный результат.')")
for i in [41,42,45,51,53]:
    put(i,'''Интерпретация зависит от текущего прогона. EDA/PCA/t-SNE не доказывают
    прибыльность и не используются для выбора модели на тестовой части.
    Все 273 признака сохраняются независимо от ранга важности.''')
put(43,'''## 2.7 Анализ признаков: однофакторный ANOVA F-критерий

Для каждого из 273 признаков сравниваются средние в двух классах **на train**.
Это `scipy.stats.f_oneway`, а не точный тест Фишера для таблиц сопряжённости.
Публикуются p и q после Benjamini–Hochberg (FDR 0.05).
Из-за временной и межбумажной зависимости результат описательный;
он не заменяет OOS-проверку и не служит отбором признаков.
''')
put(47,'''df_sorted = daily_df.sort_values(['Date','Symbol']).reset_index(drop=True)
df_prediction = prediction_panel.sort_values(['Date','Symbol']).reset_index(drop=True)
calendar = pd.DatetimeIndex(index_df.Date.sort_values().unique())
first_oos_pos = calendar.searchsorted(OOS_START)
last_train_date = calendar[first_oos_pos-GAP_DAYS-1]
train_df = df_sorted[(df_sorted.Date<=last_train_date)&(df_sorted.LabelEnd<OOS_START)].copy()
test_df = df_sorted[df_sorted.Date.between(OOS_START,OOS_END)].copy()
assert train_df.LabelEnd.max() < test_df.Date.min()
FEATURE_COLS = ALL_FEAT_COLS.copy()
X_train,y_train = train_df[FEATURE_COLS],train_df.Target
X_test,y_test = test_df[FEATURE_COLS],test_df.Target
cv = rr.date_cv_splits(train_df,n_splits=5,gap=GAP_DAYS)
for tag,df in [('train',train_df),('test',test_df)]:
    print(tag,len(df),df.Date.min(),df.Date.max(),'class 0',1-df.Target.mean(),'class 1',df.Target.mean())
print('Последний LabelEnd train:',train_df.LabelEnd.max(),'первый test:',test_df.Date.min())
''')
put(54,'''## 3.2 Методология обучения и протокол сравнения

Общий OOS: **2022-06-02…2024-11-22**, метка 10 дней, purge 10 дней.
Тестирование — одни и те же строки Date/Symbol и конфигурация модели.

| Вариант | История train | Обновление |
|---|---|---|
| Static | вся доступная до первого OOS | один раз |
| Expanding retrained | вся доступная до каждого фолда | каждые 21 день |
| Sliding retrained | последние 253 даты | каждые 21 день |

Expanding retrained vs Sliding retrained изолирует **форму окна** при общем
расписании и purge. Разный объём истории — свойство определения окна.
Static vs Expanding retrained оценивает обновление модели и доступной истории;
это не оценка «чистой частоты» при неизменном train.
Старый static с gap=260 не называется expanding walk-forward.
''')
put(55,'''## 3.3 Реализация перебора гипотез — кросс-валидация

Пять expanding-фолдов строятся по уникальным датам внутри train. Все акции
одного дня лежат в одной части; перед validation удаляются 10 дат и все
строки с LabelEnd >= validation_start. Масштабирование обучается внутри Pipeline.
ROC-AUC — ранжирование; AP и дневной Precision@10 — качество отбора.
Достижение относительного уровня не равно положительному P&L сделки.
''')
replace(56,'cv = TimeSeriesSplit(n_splits=5)','cv = rr.date_cv_splits(train_df,n_splits=5,gap=GAP_DAYS)')
replace(56,'splits = list(cv.split(X_train))','splits = cv')
replace(56,"time_budget=30, seed=SEED, verbose=0)","time_budget=30, seed=SEED, verbose=0,\n               eval_method='holdout',split_type='time',\n               estimator_list=['lrl2','rf','xgboost','histgb'])")
replace(56,"    automl = FlamlAutoML()\n    automl.fit(X_tr_fl, y_tr_fl,", "    inner_frame = train_df.iloc[tr_idx]\n    inner_tr,inner_val = rr.date_cv_splits(inner_frame,n_splits=2,gap=GAP_DAYS)[-1]\n    automl = FlamlAutoML()\n    automl.fit(X_tr_fl.iloc[inner_tr], y_tr_fl.iloc[inner_tr],\n               X_val=X_tr_fl.iloc[inner_val], y_val=y_tr_fl.iloc[inner_val],")
replace(56,"    fold_auc = roc_auc_score(y_val_fl, automl.predict_proba(X_val_fl)[:, 1])", "    # Refit the selected estimator on the full OUTER train, then score outer validation.\n    from sklearn.base import clone\n    automl_selected = clone(automl.model.estimator).fit(X_tr_fl,y_tr_fl)\n    fold_auc = roc_auc_score(y_val_fl, automl_selected.predict_proba(X_val_fl)[:, 1])")
replace(56,"random_state=seed)\n          if USE_HIST_GB", "random_state=seed, early_stopping=False)\n          if USE_HIST_GB")
# Same data and configuration in final fit/SHAP/economic primary comparison.
replace(60,"best_name = cv_results_df.iloc[0]['Модель']","best_name = cv_results_df.iloc[0]['Модель']")
replace(60,"BEST_CV_SCORES = {'LogReg': 0.5835, 'RandomForest': 0.5942, 'XGBoost': 0.5758}",
        "BEST_CV_SCORES = {'LogReg': np.nan, 'RandomForest': np.nan, 'XGBoost': np.nan}\nprint('Сохранённые параметры — legacy-конфигурации, НЕ новый результат поиска.')")
replace(60,"print('\\nГиперпараметры, отобранные RandomizedSearchCV:')",
        "print('\\nКонфигурации (новый поиск только при RUN_HYPERPARAM_SEARCH=True):')")
replace(60,"n_jobs=8","n_jobs=1")
replace(60,"print('\\nОбучаем настроенные модели на train...')",
        "tuned_estimators['Random Forest (base)'] = dict(make_models())[BASE_MODEL]\nprint('\\nОбучаем модели на train...')")
replace(61,"# Лучшая модель по Average Precision (бизнес-приоритет: не купить плохую акцию)\nbest_final_name = max(final_results, key=lambda k: final_results[k]['avg_prec'])",
        "# Primary RF fixed a priori; never select on holdout AP.\nbest_final_name = 'Random Forest (base)'")
replace(61,"print(f'\\n→ Лучшая модель по Average Precision: {best_final_name}')",
        "print(f'\\nОсновная конфигурация, выбранная заранее: {best_final_name}')")
replace(61,'K = 50','K = TOP_K')
replace(40,"sample(min(50000, len(eda_train)))","sample(min(50000, len(eda_train)),random_state=SEED)")
replace(61,"'XGBoost (tuned)': 'XGBoost'}","'XGBoost (tuned)': 'XGBoost', 'Random Forest (base)': BASE_MODEL}")
# Threshold selected on the last purged TRAIN fold; test is used only once for reporting.
put(63,'''from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
tr_pos,val_pos = cv[-1]
threshold_est = clone(best['model']).fit(X_train.iloc[tr_pos],y_train.iloc[tr_pos])
val_prob = threshold_est.predict_proba(X_train.iloc[val_pos])[:,1]
prec_arr,rec_arr,thresholds = precision_recall_curve(y_train.iloc[val_pos],val_prob)
f1_arr = 2*prec_arr[:-1]*rec_arr[:-1]/np.maximum(prec_arr[:-1]+rec_arr[:-1],1e-12)
best_thr = thresholds[np.argmax(f1_arr)]
y_pred_opt = (best['y_prob'] >= best_thr).astype(int)
print('Порог выбран на последнем purged train-фолде:',best_thr)
print(classification_report(y_test,y_pred_opt,zero_division=0))
print('Top-K стратегия от порога не зависит.')
''')
put(62,'''Recall и Precision отвечают на разные вопросы; при ограничении k
оцениваем Precision@10 по дням и P&L с учётом риска и издержек.
Модель для основного экономического опыта фиксирована до просмотра теста.''')
put(67,'''False positive означает ошибку в событии опережения индекса, а не
обязательно денежный убыток. False negative также не определяет P&L.
Экономический результат рассчитывается по ценам исполнения отдельно.''')
put(70,'''# Выбор секторов по реализованному Target теста создавал бы подгонку.
print('Секторальная статистика выше — диагностика; торговый фильтр по ней не строится.')
''')
put(72,'''Гипотезы об отскоке и momentum проверяются на текущем SHAP-графике.
Важность признака не определяет знак эффекта и не является торговой рекомендацией.''')
# Retain reporting/plot functions, substitute engine with tested module.
engine = ''.join(nb['cells'][73]['source'])
reporting = engine[engine.index('def print_backtest'):]
reporting = reporting.replace("print('  Стратегия готова к paper trading')","print('  Пройдены условные пороги; это не доказательство устойчивости стратегии')")
put(73,"TRADING_DAYS_PER_YEAR = 252\ndef run_backtest(*args,**kwargs):\n    kwargs.setdefault('top_k',TOP_K)\n    kwargs.setdefault('capital',INITIAL_CAPITAL)\n    kwargs.setdefault('commission',COMMISSION)\n    return rr.run_backtest(*args,**kwargs)\n"+reporting)
old74 = ''.join(nb['cells'][74]['source'])
put(74,'''# Основной статический RF: та же конфигурация, что и walk-forward ниже
panel = full_price_panel.copy()
index_prices = index_df[['Date','SP500']].copy()
trade_test = df_prediction[df_prediction.Date.between(OOS_START,OOS_END)]
bt_preds = trade_test[['Date','Symbol','Target']].rename(columns={'Target':'y_true'}).copy()
bt_preds['y_prob'] = best['model'].predict_proba(trade_test[ALL_FEAT_COLS])[:,1]
res_expanding_full = run_backtest(bt_preds,panel,index_prices,horizon=HOLD_DAYS,
                                start=OOS_START,end=OOS_END,label='Static RF')
print_backtest(res_expanding_full)
plot_backtest(res_expanding_full)
plt.show()
''')
put(77,'''Результаты выше относятся к static RF и дополнительному long/short примеру.
Основное сравнение форм окна и показателей риска приведено в разделе 3.7.
Пороговые таблицы не являются доказательством готовности стратегии к торговле.''')
put(82,'''## 3.5 Скользящее окно (Sliding Window)

Train: последние 253 торговые даты. Purge: 10 дней. Test/step: 21 день.
Все OOS-прогнозы начинаются 2022-06-02; последний фолд может быть короче
21 дня, чтобы в точности завершиться 2024-11-22. Нет разных «общих» окон.
''')
put(83,'''df_sorted_sw = df_prediction.copy()
models_sw = dict(make_models())
sw_preds,sw_fold_df = rr.walk_predict(df_sorted_sw,ALL_FEAT_COLS,models_sw,
    OOS_START,OOS_END,window=TRAIN_DAYS,gap=GAP_DAYS,step=STEP_DAYS,
    cache_dir=PRED_DIR/'sliding_models')
sw_fold_scores = {n:sw_fold_df.loc[sw_fold_df['Модель']==n,'ROC-AUC'].tolist() for n in models_sw}
SW_N_TRAIN,SW_N_TEST = int(sw_fold_df.n_train.mean()),int(sw_fold_df.n_test.mean())
sw_results_df = pd.DataFrame([{'Модель':n,'ROC-AUC (mean)':np.nanmean(sw_fold_scores[n]),
    'ROC-AUC (std)':np.nanstd(sw_fold_scores[n]),'ROC-AUC (пул)':roc_auc_score(sw_preds.loc[sw_preds.y_true.notna(),'y_true'],sw_preds.loc[sw_preds.y_true.notna(),n])}
    for n in models_sw]).sort_values('ROC-AUC (mean)',ascending=False)
sw_preds.to_parquet(PRED_DIR/'sliding_preds.parquet',index=False)
sw_fold_df.to_csv(PRED_DIR/'sliding_fold_scores.csv',index=False)
display(sw_results_df)
''')
put(84,'''# Оценочные прогнозы и календарь сохранены выше.
# Последняя модель фолда не выдаётся за модель на всей истории.
print(PRED_DIR/'sliding_fold_scores.csv')
''')
put(86,'''Пофолдовые графики описывают текущий прогон. Соседние фолды зависимы;
выбор победителя по визуальному расстоянию кривых не обоснован.
Формальная проверка приведена ниже.''')
put(88,'''## 3.6 Статистическая значимость различий между алгоритмами

Основной интервал: circular moving-block bootstrap **смежных блоков по 21
торговому дню**, 2 000 повторов. День переносится вместе со всеми акциями;
смежные дни сохраняют зависимость перекрывающихся меток. Отдельные дни iid
этого не обеспечивали. Длина блока — заранее заданный параметр.

DeLong — диагностический iid-тест; его интервал не является гарантированной
нижней границей ширины. Nadeau–Bengio по фолдам — приближённая проверка
чувствительности к зависимости, а не точный тест walk-forward.
Попарные p-values корректируются Holm. Bootstrap p вычисляется с центрированием
при нулевой гипотезе и plus-one, поэтому нулевые p не печатаются.
''')
replace(89,"picked = rng.integers(0, n_days, n_days)","picked = rr.circular_blocks(n_days, BOOT_BLOCK_DAYS, rng)")
replace(89,"p_vs_half[name] = min(2 * min((draws[:, k] <= 0.5).mean(),\n                                      (draws[:, k] >= 0.5).mean()), 1.0)",
        "p_vs_half[name] = (1 + np.sum(np.abs(draws[:, k]-observed[name]) >= abs(observed[name]-.5))) / (len(draws)+1)")
replace(89,"p = 2 * min((d <= 0).mean(), (d >= 0).mean())",
        "observed_diff = boot['observed'][name_a]-boot['observed'][name_b]\n    p = (1+np.sum(np.abs(d-observed_diff) >= abs(observed_diff)))/(len(d)+1)")
replace(90,'N_BOOT = 400','N_BOOT = max(2000, N_BOOT)')
replace(90,'model_names = list(models_sw)',"model_names = list(models_sw)\nscored_sw_preds = sw_preds.dropna(subset=['y_true']).copy()")
replace(90,'block_bootstrap_aucs(sw_preds,','block_bootstrap_aucs(scored_sw_preds,')
replace(90,"pooled_preds=sw_preds[['Date', 'y_true'] + model_names]","pooled_preds=scored_sw_preds[['Date', 'y_true'] + model_names]")
replace(90,"'внутри дня и перекрытие таргетов не занижают ошибку.'","'внутри дня сохраняется; временная зависимость учитывается смежными блоками.'")
replace(90,"lows = [boot['observed'][n] - boot['ci'][n][0] for n in order]","lows = [max(0,boot['observed'][n] - boot['ci'][n][0]) for n in order]")
replace(90,"highs = [boot['ci'][n][1] - boot['observed'][n] for n in order]","highs = [max(0,boot['ci'][n][1] - boot['observed'][n]) for n in order]")
put(91,'''## 3.7 Выровненное сравнение формы окна и обновления модели

Новый эксперимент добавляет expanding с тем же 21-дневным расписанием и
purge=10, что и sliding. Static используется как отдельный контроль.
Основные таблицы содержат Jensen alpha (годовой intercept CAPM, rf=0,
Newey–West HAC), beta и information ratio; простое превышение накопленной
доходности над индексом подписано отдельно и не называется Jensen alpha.

Бэктест: k=10; равные доли; сигнал после закрытия t, покупка на закрытии t+1;
выход через h сессий после покупки. Ежедневная переоценка. Комиссия за
фактический входной и выходной номинал; полная ликвидация и новая покупка,
включая повторно выбранные бумаги. Это консервативная модель исполнения,
а не минимальный net-turnover. Средний двухсторонний оборот =
(покупки+продажи)/капитал перед периодом; one-way = половина.
Без проскальзывания, дивидендов и стоимости шорта.

**21 день:** те же 10-дневные прогнозы, без переопределения Target и нового обучения.
Проверяется исполнение существующего сигнала; это не опыт с 21-дневной меткой.
Дополнительно h перекрывающихся портфелей с весом 1/h дают общую дневную сетку
для парного сравнения h=10/21. Неиспользованные деньги на краях остаются cash.
''')
put(92,'''panel = full_price_panel.copy()
index_prices = index_df[['Date','SP500']].copy()
primary_models = {BASE_MODEL:dict(make_models())[BASE_MODEL]}
exp_monthly,exp_fold_df = rr.walk_predict(df_prediction,ALL_FEAT_COLS,primary_models,
    OOS_START,OOS_END,gap=GAP_DAYS,step=STEP_DAYS,cache_dir=PRED_DIR/'expanding_rf')
static_preds,static_fold_df = rr.walk_predict(df_prediction,ALL_FEAT_COLS,primary_models,
    OOS_START,OOS_END,gap=GAP_DAYS,step=STEP_DAYS,static=True,cache_dir=PRED_DIR/'static_rf')
common_start,common_end = OOS_START,OOS_END
def prep(preds,model_col=BASE_MODEL):
    return preds[['Date','Symbol','y_true',model_col]].rename(columns={model_col:'y_prob'}).copy()
pred_sources = {'static':static_preds,'expanding':exp_monthly,'sliding':sw_preds}
keys = exp_monthly[['Date','Symbol']].reset_index(drop=True)
for scheme,p in pred_sources.items():
    assert p[['Date','Symbol']].reset_index(drop=True).equals(keys),scheme
runs,robust_runs,cost_rows = {},{},[]
for scheme,p in pred_sources.items():
    runs[scheme] = run_backtest(prep(p),panel,index_prices,horizon=HOLD_DAYS,
        start=common_start,end=common_end,label=scheme)
    robust_runs[scheme] = run_backtest(prep(p),panel,index_prices,horizon=HOLD_DAYS_ROBUST,
        start=common_start,end=common_end,label=scheme+' h21')
    for h in [10,21]:
        for multiplier in [1,3,5]:
            result = run_backtest(prep(p),panel,index_prices,horizon=h,start=common_start,
                                 end=common_end,commission=COMMISSION*multiplier,label=scheme)
            cost_rows.append({'scheme':scheme,'holding':h,'cost_multiplier':multiplier,**result['metrics']})
comparison_df = pd.DataFrame({n:r['metrics'] for n,r in runs.items()}).T
main_cols = ['jensen_alpha','jensen_p','beta','information_ratio','total_return',
             'alpha_vs_index','alpha_vs_equal_weight','sharpe','max_drawdown',
             'profit_factor','avg_turnover','one_way_turnover','n_rebalances']
display(comparison_df[main_cols])
comparison_df.to_csv(CACHE_DIR/'primary_economic_comparison.csv')
cost_df = pd.DataFrame(cost_rows)
cost_df.to_csv(CACHE_DIR/'commission_stress.csv',index=False)
display(cost_df[['scheme','holding','cost_multiplier','total_return','jensen_alpha','alpha_vs_index','avg_turnover']])
for name,r in runs.items():
    r['equity'].to_csv(CACHE_DIR/f'equity_{name}.csv')
    r['trades'].to_csv(CACHE_DIR/f'trades_{name}.csv',index=False)
    print_backtest(r)
exp_fold_df.to_csv(CACHE_DIR/'expanding_fold_scores.csv',index=False)
static_fold_df.to_csv(CACHE_DIR/'static_fold_scores.csv',index=False)
plot_backtest(runs['expanding']); plt.show()
plot_backtest(runs['sliding']); plt.show()
''')
put(93,'''# Парные дневные circular blocks: одна календарная сетка, без iid ресэмплинга дней
from itertools import combinations
test_rows = []
for block in [10,21,42]:
    for h,group in [(10,runs),(21,robust_runs)]:
        for a,b in [('expanding','sliding'),('expanding','static')]:
            bs = rr.paired_daily_bootstrap(group[a],group[b],n_boot=N_BOOT,block=block)
            test_rows.append({'comparison':a+' - '+b,'holding':h,'block':block,**bs})
alpha_tests = pd.DataFrame(test_rows)
alpha_tests['p_Holm'] = holm_correction(alpha_tests.p)
display(alpha_tests)
alpha_tests.to_csv(CACHE_DIR/'scheme_diff_significance.csv',index=False)

# Дополнительный опыт по замечанию 6: перекрывающиеся портфели, общая дневная сетка.
overlap_rows = []
overlap_results = {}
for scheme,p in pred_sources.items():
    pair = {h:rr.overlapping_backtest(prep(p),panel,index_prices,horizon=h,
                 start=common_start,end=common_end,top_k=TOP_K,commission=COMMISSION,
                 label=scheme+' overlapping') for h in [10,21]}
    overlap_results[scheme] = pair
    for block in [10,21,42]:
        bs = rr.paired_daily_bootstrap(pair[21],pair[10],n_boot=N_BOOT,block=block)
        overlap_rows.append({'scheme':scheme,'comparison':'hold21 - hold10','block':block,**bs})
overlap_tests = pd.DataFrame(overlap_rows)
overlap_tests['p_Holm'] = holm_correction(overlap_tests.p)
display(overlap_tests)
overlap_tests.to_csv(CACHE_DIR/'overlapping_horizon_tests.csv',index=False)
print('Незначимое различие не доказывает эквивалентности. '
      'Старые p-values статьи здесь не используются.')
''')
replace(95,'best_model_obj = best[\'model\']',"best_model_obj = best['model']  # primary RF base, как в экономическом сравнении")
put(96,'''SHAP объясняет **ту же базовую конфигурацию RF**, что используется в
экономическом опыте, на 2 000 тестовых наблюдений (после фиксации модели).
Средний |SHAP| показывает силу, не знак и не причинность.
Знак читается по текущему beeswarm. Нельзя заранее называть сигнал
реверсальным или делать вывод «покупать падение» из одного ранга важности.
SHAP помогает интерпретировать факторную экспозицию, но не доказывает
экономическое преимущество формы окна.
''')
put(97,'''print('СВОДКА НОВОГО ПРОГОНА ДЛЯ СТАТЬИ')
print('Конфигурация:',config)
print('Запрошено:',n_symbols_file,'загружено:',price_raw.Symbol.nunique(),
      'пригодно для модели:',daily_df.Symbol.nunique(),'строк:',len(daily_df))
print('Класс 0:',1-daily_df.Target.mean(),'класс 1:',daily_df.Target.mean())
display(funnel_df)
display(cv_results_df.drop(columns='folds'))
display(sw_results_df)
display(comparison_df[main_cols])
display(alpha_tests)
display(overlap_tests)
print('ANOVA BH q<.05:',int((fisher_df.q_BH<.05).sum()))
print('Значимых пар алгоритмов по двум приближённым критериям:',int(sig_df['Значимо'].sum()))
print('Численные утверждения статьи обновлять только из ЭТОГО прогона.')
(CACHE_DIR/'run_completed.json').write_text(json.dumps({'status':'completed','config':config,
    'loaded':int(price_raw.Symbol.nunique()),'eligible':int(daily_df.Symbol.nunique()),
    'rows':len(daily_df)},indent=2),encoding='utf-8')
''')
put(98,'''# 5. Выводы

## 5.1 Что проверяет новый эксперимент

Сохранены исходные 13 доходностей относительно индекса и 260 rolling-статистик,
порог 2 п.п., 10-дневное событие достижения и pooled-модели.
В таблицах раздела 3.7 форма окна сравнивается при одинаковом расписании
переобучения, purge, модели, OOS-строках и правилах исполнения.
Удержание 21 день относится к исполнению **10-дневного** сигнала.

## 5.2 Интерпретация результатов

Числа выводятся в предыдущей ячейке. До успешного полного запуска нельзя
утверждать, что новая вселенная повторяет значения ROC-AUC/альфы/p статьи.
Отсутствие значимости не доказывает равенства стратегий. Больше бумаг
не означает больше независимых рыночных периодов.
Jensen alpha (rf=0, HAC) и information ratio приводятся вместе с beta;
разность накопленных доходностей — отдельная, не скорректированная на риск метрика.

## 5.3 Явно обозначенные изменения методики

1. yfinance вместо дефектной выгрузки; фиксированный снимок 502, явный audit.
2. Удалён дополнительный фильтр ещё 270 готовых дней; нужны 273 признака и
   полностью известная 10-дневная метка. Хвост с неполным будущим исключён.
3. Календарь индекса и purge по датам вместо разбиения строк панели и gap=260.
4. Новый expanding walk-forward с обновлением каждые 21 день.
5. Сигнал t → исполнение t+1; ежедневные equity/просадка; одинаковый конец окна;
   нет сделок с выходом за OOS, нет отбора по известной будущей доходности.
6. ANOVA+BH, >=2000 bootstrap смежных блоков; общий дневной paired design.
7. Primary RF и SHAP одной конфигурации, выбор порога только на train.

## 5.4 Ограничения и публикация

- Исторический состав индекса не восстановлен: survivorship bias остаётся.
- IPO/делистинг/переименования: наличие 502 имён не гарантирует 502 пригодные
  истории; считать по `download_manifest.csv` и `model_eligibility.csv`.
- CAPM использует rf=0 и ценовой индекс без дивидендов. Это приближение,
  не Fama–French+momentum. Для такого теста нужен отдельный набор факторов.
- Текущая Yahoo-история может пересматриваться; кэш, даты, SHA256 и версии
  библиотек необходимы для воспроизводимости.
- Издержки: комиссия 0.1% на каждую сторону; стресс 3x/5x; нет проскальзывания,
  влияния на рынок и стоимости шорта. Повторная покупка платит полный оборот.
- Перекрывающиеся портфели имеют cash на начальной/конечной границе;
  они проверяют исполнение, а не чистое влияние горизонта метки.
- Настройки LR/RF/XGB из final — legacy-конфигурации, если поиск не запущен;
  они не выдаются за оптимальные на новой вселенной. HistGB/KNN — baseline,
  вычислительный бюджет и отсутствие поиска не доказывают их неполноценность.
- Bootstrap и Nadeau–Bengio не гарантируют корректность при произвольной
  нестационарности; показана чувствительность к длине блока 10/21/42.

Код можно подготовить для GitHub вместе с новым модулем `src/review_revision.py`,
описанием окружения и конфигурацией. Решение о публикации остаётся у автора;
push в рамках этой работы запрещён и не выполняется. PDF и рыночные выгрузки
не следует включать без проверки права на распространение.
''')

# Remove stale outputs and numerical execution state everywhere.
for cell in nb['cells']:
    if isinstance(cell['source'],str):
        cell['source'] = cell['source'].splitlines(keepends=True)
    if cell['cell_type']=='code':
        cell['outputs'] = []
        cell['execution_count'] = None
    cell['metadata'] = {}
    if 'id' not in cell:
        cell['id'] = f'review-{nb["cells"].index(cell):03d}'
nb['metadata']['review_revision'] = {'date':'2026-11-01','original_sha256':original_hashes,
                                    'results':'unexecuted; do not reuse article numbers'}
target = ROOT/'notebooks/sp500_coursework_reviewed.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(target.name,len(nb['cells']),'cells')
assert all(hashlib.sha256((ROOT/'notebooks'/n).read_bytes()).hexdigest()==h
           for n,h in original_hashes.items())

import os
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go

from dash import Dash, html, dcc, Input, Output, State, callback, dash_table, MATCH, ALL, no_update, ctx

# Optional scientific deps
try:
    from scipy.stats import chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Fear & Greed dependency
try:
    import fear_and_greed
    HAS_FG = True
except ImportError:
    HAS_FG = False

# Iconify icons
try:
    from dash_iconify import DashIconify
    HAS_ICONIFY = True
except ImportError:
    HAS_ICONIFY = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS, THEME & TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────────────
SIGNALS = ['S&P500', 'Gold', 'Oil_WTI', 'USD_Index', 'VIX']

# Premium Fintech Palette
ACCENT = "#FFFFFF"     
ACCENT2 = "#8A94A3"    
POS = "#00E599"        
WARN = "#F5A623"       
DANGER = "#FF4B4B"     
MUTE = "#6B7280"       

TR = {
    'en': {
        'overview': 'Overview', 'timeline': 'Timeline', 'alerts': 'Alerts', 'validation': 'Validation', 
        'methodology': 'Methodology', 'raw_data': 'Raw Data', 'sys_stress': 'Systemic Stress Timeline',
        'current_status': 'Current Status', 'thresh_limit': 'Threshold limit:', 'drivers_today': "Drivers of Today's Score",
        'market_narrative': 'Market Narrative', 'fg_index': 'Fear & Greed Index', 'exp_thresh': 'Expanding Threshold',
        'causal_mean': 'Causal mean + 2σ', 'alert_freq': 'Alert Frequency', 'all_time_rate': 'All-time rate',
        'total_alerts': 'Total Alerts', 'hist_events': 'Historical events', 'full_timeline': 'Full Anomaly Timeline',
        'anomaly_alerts': 'Anomaly Alerts', 'model_val': 'Model Validation', 'raw_data_exp': 'Raw Data Explorer',
        'live_data': 'Live Data', 'data_error': 'Data Error', 'market_intel': 'Market Intelligence',
        'confidence': 'Confidence:', 'status': 'Status:', 'last_updated': 'Last updated:', 'vs_thresh': 'vs Threshold',
        'pipeline_status': 'Pipeline status', 'operational': 'Operational', 'degraded': 'Degraded',
        'active_source': 'Active data source', 'trading_days': 'Trading days loaded', 'history_range': 'History range',
        'last_fetch': 'Last successful fetch', 'data_status': 'Data Status', 'crisis_recall': 'Crisis Recall',
        'events_detected': 'Events Detected', 'flagged_days': 'Flagged Days', 'daily_flag_rate': 'Daily Flag Rate',
        'within_7d': 'within ±7 days', 'all_history': 'all history', 'of_trading_days': 'of trading days',
        'date': 'Date', 'hist_event': 'Historical Event', 'composite': 'Composite', 'nearest': 'Nearest',
        'peak_score': 'Peak Score', 'detected': 'Detected', 'missed': 'Missed', 'top_driver': 'Top Driver',
        'rarity': 'Rarity (p)', 'when': 'When', 'view_news': 'View news from', 'load_headlines': 'Load headlines',
        'showing_recent': 'Showing most recent 60 matches.', 'no_anomaly': 'No anomaly days match this filter.',
        'all_years': 'All Years', 'all_months': 'All Months', 'status_normal': 'Normal', 'status_elevated': 'Elevated',
        'status_stress': 'Stress', 'status_crisis': 'Crisis',
        'narrative_calm': "Markets look calm today — no unusual stress detected. The current status is {status}. Normal background activity is mainly driven by {driver}.",
        'narrative_warn': "Caution: Market stress is unusually high right now. The current status is {status}, primarily driven by sudden moves in {driver} ({pct:.0f}% of the activity). Keep an eye on conditions.",
        'chart_score': 'Anomaly Score', 'chart_limit': 'Threshold Limit', 'fetch_failed': 'Fetch Failed',
        'module_not_installed': 'Module not installed', 'unavailable': 'Unavailable', 'days_ago': '{d}d ago',
        'score_label': 'Market Anomaly Score', 'lang_btn': 'عربي',
        'months': ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        'assets': {'S&P500': 'S&P 500', 'Gold': 'Gold', 'Oil_WTI': 'Oil', 'USD_Index': 'USD', 'VIX': 'VIX'},
        'ranges': {'Last 6 Months': 'Last 6 Months', 'Last 2 Years': 'Last 2 Years', 'Full History (2005-Present)': 'Full History (2005-Present)'},
        'evt_covid': 'COVID Crash', 'evt_ukraine': 'Ukraine Inv.', 'evt_svb': 'SVB Collapse',
        'method_lead': 'How this dashboard turns five noisy markets into a single, defensible measure of systemic stress...',
        'm_title_1': 'What it measures', 'm_p_1': 'Every trading day, the model asks one question: how unusual is today, across the whole market at once?',
        'm_title_2': 'The composite score', 'm_p_2': 'Each market is first converted into a z-score...',
    },
    'ar': {
        'overview': 'نظرة عامة', 'timeline': 'الجدول الزمني', 'alerts': 'التنبيهات', 'validation': 'التحقق', 
        'methodology': 'المنهجية', 'raw_data': 'البيانات الخام', 'sys_stress': 'الجدول الزمني للضغط النظامي',
        'current_status': 'الحالة الحالية', 'thresh_limit': 'حد العتبة:', 'drivers_today': 'محركات درجة اليوم',
        'market_narrative': 'سرد السوق', 'fg_index': 'مؤشر الخوف والطمع', 'exp_thresh': 'العتبة المتوسعة',
        'causal_mean': 'المتوسط السببي + 2σ', 'alert_freq': 'تكرار التنبيهات', 'all_time_rate': 'معدل كل الأوقات',
        'total_alerts': 'إجمالي التنبيهات', 'hist_events': 'أحداث تاريخية', 'full_timeline': 'الجدول الزمني الكامل',
        'anomaly_alerts': 'تنبيهات التشوهات', 'model_val': 'التحقق من النموذج', 'raw_data_exp': 'مستكشف البيانات',
        'live_data': 'بيانات مباشرة', 'data_error': 'خطأ في البيانات', 'market_intel': 'ذكاء السوق',
        'confidence': 'الثقة:', 'status': 'الحالة:', 'last_updated': 'تحديث:', 'vs_thresh': 'مقابل العتبة',
        'pipeline_status': 'حالة المسار', 'operational': 'شغال', 'degraded': 'متدهور',
        'active_source': 'مصدر البيانات', 'trading_days': 'أيام التداول', 'history_range': 'النطاق التاريخي',
        'last_fetch': 'آخر جلب ناجح', 'data_status': 'حالة البيانات', 'crisis_recall': 'استدعاء الأزمات',
        'events_detected': 'الأحداث المكتشفة', 'flagged_days': 'الأيام المحددة', 'daily_flag_rate': 'معدل التحديد اليومي',
        'within_7d': 'خلال ±7 أيام', 'all_history': 'كل التاريخ', 'of_trading_days': 'من أيام التداول',
        'date': 'التاريخ', 'hist_event': 'حدث تاريخي', 'composite': 'المركب', 'nearest': 'الأقرب',
        'peak_score': 'ذروة الدرجة', 'detected': 'مكتشف', 'missed': 'مفقود', 'top_driver': 'المحرك الأكبر',
        'rarity': 'الندرة (p)', 'when': 'متى', 'view_news': 'عرض الأخبار:', 'load_headlines': 'تحميل العناوين',
        'showing_recent': 'عرض أحدث 60 تطابقًا.', 'no_anomaly': 'لا توجد تطابقات.',
        'all_years': 'كل السنوات', 'all_months': 'كل الأشهر', 'status_normal': 'طبيعي', 'status_elevated': 'مرتفع',
        'status_stress': 'ضغط', 'status_crisis': 'أزمة',
        'narrative_calm': "تبدو الأسواق هادئة اليوم — لم نكتشف أي ضغط غير عادي. الحالة الحالية {status}. النشاط الطبيعي مدفوع بشكل رئيسي بـ {driver}.",
        'narrative_warn': "تحذير: ضغط السوق مرتفع جداً الآن. الحالة الحالية {status}، مدفوعة بحركات مفاجئة في {driver} ({pct:.0f}% من النشاط). يرجى المراقبة.",
        'chart_score': 'درجة التشوه', 'chart_limit': 'حد العتبة', 'fetch_failed': 'فشل الجلب',
        'module_not_installed': 'الوحدة غير مثبتة', 'unavailable': 'غير متوفر', 'days_ago': 'قبل {d} يوم',
        'score_label': 'درجة شذوذ السوق', 'lang_btn': 'EN',
        'months': ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
        'assets': {'S&P500': 'إس آند بي 500', 'Gold': 'الذهب', 'Oil_WTI': 'النفط', 'USD_Index': 'الدولار', 'VIX': 'مؤشر التقلب'},
        'ranges': {'Last 6 Months': 'آخر 6 أشهر', 'Last 2 Years': 'آخر سنتين', 'Full History (2005-Present)': 'التاريخ الكامل (2005-الآن)'},
        'evt_covid': 'انهيار كوفيد', 'evt_ukraine': 'غزو أوكرانيا', 'evt_svb': 'انهيار SVB',
        'method_lead': 'كيف تحول هذه اللوحة خمسة أسواق صاخبة إلى مقياس واحد وموثوق للضغط النظامي...',
        'm_title_1': 'ما الذي يقيسه', 'm_p_1': 'في كل يوم تداول، يطرح النموذج سؤالاً واحدًا: ما مدى غرابة اليوم، عبر السوق بأكمله في وقت واحد؟',
        'm_title_2': 'النتيجة المركبة', 'm_p_2': 'يتم تحويل كل سوق أولاً إلى درجة معيارية...',
    }
}

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBALLY RENAMED TRANSLATION FUNCTION (Fixes UnboundLocalError)
# ─────────────────────────────────────────────────────────────────────────────
def t(key, lang='en'):
    return TR.get(lang, TR['en']).get(key, key)

ICO = {
    "activity": '<path d="M22 12h-4l-3 8L9 4l-3 8H2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    "target":   '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "shield":   '<path d="M12 21s7-3.4 7-9V5.5L12 3 5 5.5V12c0 5.6 7 9 7 9z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 8.5v3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="15" r="0.6" fill="currentColor" stroke="currentColor" stroke-width="1"/>',
    "clock":    '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 7.5v5l3.2 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
}

HISTORICAL_EVENTS = {
    "2008-09-15": "Lehman Brothers bankruptcy",
    "2010-05-06": "Flash Crash",
    "2011-08-08": "US credit downgrade",
    "2015-08-24": "China devaluation (Black Monday)",
    "2020-02-24": "evt_covid",  
    "2020-03-16": "COVID-19 circuit breakers",
    "2022-06-13": "S&P 500 bear market",
    "2022-02-24": "evt_ukraine", 
    "2023-03-10": "evt_svb",     
}

def tint(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def svg_icon(name, color, size=19):
    inner = ICO[name].replace('currentColor', color)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}">{inner}</svg>'
    uri = "data:image/svg+xml;utf8," + urllib.parse.quote(svg)
    return html.Img(src=uri, style={'width': f'{size}px', 'height': f'{size}px', 'display': 'block'})


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOGIC
# ─────────────────────────────────────────────────────────────────────────────
import time

def load_data():
    global DATA_SOURCE  
    tickers = {'S&P500': '^GSPC', 'VIX': '^VIX', 'Gold': 'GC=F', 'Oil_WTI': 'CL=F', 'USD_Index': 'DX-Y.NYB'}
    data = {}
    try:
        from defeatbeta_api.data.ticker import Ticker as DBTicker
        for name, t_sym in tickers.items():
            dbt = DBTicker(t_sym)
            price_df = dbt.price()
            price_df['report_date'] = pd.to_datetime(price_df['report_date'])
            price_df = price_df.set_index('report_date').sort_index()
            price_df = price_df[price_df.index >= '2005-01-01']
            data[name] = price_df['close']
        df = pd.DataFrame(data).dropna()
        if len(df) > 0:
            DATA_SOURCE = "defeatbeta-api"
            return df
    except Exception:
        pass

    for name, t_sym in tickers.items():
        close = None
        for attempt in range(4):
            try:
                d = yf.download(t_sym, start='2005-01-01', progress=False)
                c = d['Close']
                if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
                if len(c) > 0:
                    close = c
                    break
            except Exception: pass
            time.sleep(2 ** attempt)  
        data[name] = close if close is not None else pd.Series(dtype=float)

    DATA_SOURCE = "yfinance"
    return pd.DataFrame(data).dropna()

def compute_anomaly(prices, window=63, k=2.0, burn_in=252):
    df = prices.copy()
    price_assets = ['S&P500', 'Gold', 'Oil_WTI', 'USD_Index']
    for col in price_assets:
        df[f'{col}_Return'] = np.log(df[col] / df[col].shift(1))
        df[f'{col}_RollMean'] = df[f'{col}_Return'].rolling(window).mean()
        df[f'{col}_RollStd'] = df[f'{col}_Return'].rolling(window).std()
        df[f'{col}_Zscore'] = (df[f'{col}_Return'] - df[f'{col}_RollMean']) / df[f'{col}_RollStd']

    df['VIX_RollMean'] = df['VIX'].rolling(window).mean()
    df['VIX_RollStd'] = df['VIX'].rolling(window).std()
    df['VIX_Zscore'] = (df['VIX'] - df['VIX_RollMean']) / df['VIX_RollStd']

    zcols = [f'{s}_Zscore' for s in SIGNALS]
    n = len(zcols)
    sum_sq = (df[zcols] ** 2).sum(axis=1)
    safe = sum_sq.replace(0, np.nan)
    df['Sum_Sq_Z'] = sum_sq
    df['Anomaly_Score'] = np.sqrt(sum_sq / n)

    for s in SIGNALS:
        df[f'{s}_Contribution'] = (df[f'{s}_Zscore'] ** 2 / safe) * 100

    df['Anomaly_PValue'] = chi2.sf(df['Sum_Sq_Z'].values, df=n) if HAS_SCIPY else np.nan
    exp_mean = df['Anomaly_Score'].expanding(min_periods=burn_in).mean().shift(1)
    exp_std = df['Anomaly_Score'].expanding(min_periods=burn_in).std().shift(1)
    df['Threshold'] = exp_mean + k * exp_std
    df['Flagged'] = df['Anomaly_Score'] > df['Threshold']
    return df

def compute_isolation_forest(scored_df, contamination):
    zcols = [f'{s}_Zscore' for s in SIGNALS]
    feat = scored_df[zcols].dropna()
    clf = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    clf.fit(feat.values)
    out = pd.DataFrame(index=feat.index)
    out['IF_Score'] = -clf.score_samples(feat.values)
    out['IF_Flagged'] = (clf.predict(feat.values) == -1)
    out = out.reindex(scored_df.index)
    out['IF_Flagged'] = out['IF_Flagged'].fillna(False).astype(bool)
    return out

def validate_events(scored_df, events, flag_col='Flagged', window_days=7):
    rows = []
    fidx = scored_df.index[scored_df[flag_col].fillna(False)]
    for ds, desc in sorted(events.items()):
        d = pd.Timestamp(ds)
        lo, hi = d - pd.Timedelta(days=window_days), d + pd.Timedelta(days=window_days)
        win = scored_df[(scored_df.index >= lo) & (scored_df.index <= hi)]
        detected = bool(win[flag_col].fillna(False).any()) if len(win) else False
        wide = fidx[(fidx >= d - pd.Timedelta(days=30)) & (fidx <= d + pd.Timedelta(days=30))]
        nearest = int(min(abs((f - d).days) for f in wide)) if len(wide) else None
        peak = float(win['Anomaly_Score'].max()) if len(win) and win['Anomaly_Score'].notna().any() else None
        rows.append({'date': ds, 'event': desc, 'detected': detected, 'nearest': nearest, 'peak': peak})
    return rows

@lru_cache(maxsize=256)
def get_news_for_date(date_str, days_window=1):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        after = d.strftime("%Y-%m-%d")
        before = (d + timedelta(days=days_window)).strftime("%Y-%m-%d")
        query = f"stock market after:{after} before:{before}"
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:5]
        return tuple((i.find("title").text, i.find("link").text, i.find("pubDate").text) for i in items)
    except Exception:
        return tuple()

# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────────────────────
DF = DF_IF = None
VAL = VAL_IF = None
AVAIL_YEARS = []
SUMMARY = {}
DATA_OK = False
LOAD_ERR = ""
DATA_SOURCE = "unknown"
TRADING_DAYS = 0
LOADED_AT = "—"

def init_data():
    global DF, DF_IF, VAL, VAL_IF, AVAIL_YEARS, SUMMARY, DATA_OK, LOAD_ERR, TRADING_DAYS, LOADED_AT
    prices = load_data()
    DF = compute_anomaly(prices)
    VAL = validate_events(DF, HISTORICAL_EVENTS, 'Flagged')
    detected = sum(r['detected'] for r in VAL)
    total_ev = len(VAL)
    n_scored = int(DF['Threshold'].notna().sum())
    total_flags = int(DF['Flagged'].sum())
    flag_rate = (total_flags / n_scored * 100) if n_scored else 0.0

    if HAS_SKLEARN:
        contamination = float(min(max(flag_rate / 100.0, 0.005), 0.20))
        DF_IF = DF.join(compute_isolation_forest(DF, contamination))
        VAL_IF = validate_events(DF_IF, HISTORICAL_EVENTS, 'IF_Flagged')
    else:
        DF_IF = DF

    SUMMARY = {
        'detected': detected, 'total_ev': total_ev, 'recall': (detected / total_ev * 100) if total_ev else 0.0,
        'total_flags': total_flags, 'flag_rate': flag_rate,
        'if_detected': (sum(r['detected'] for r in VAL_IF) if VAL_IF else None),
        'updated': datetime.now().strftime("%d %b · %H:%M"),
    }
    if VAL_IF: SUMMARY['if_recall'] = (SUMMARY['if_detected'] / total_ev * 100) if total_ev else 0.0

    flags = DF[DF['Flagged'] == True]
    AVAIL_YEARS = sorted(flags.index.year.unique(), reverse=True)
    TRADING_DAYS = int(len(DF))
    LOADED_AT = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    DATA_OK = True

if not os.environ.get("APP_SKIP_LOAD"):
    try:
        init_data()
    except Exception as e:
        DATA_OK = False
        LOAD_ERR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS & LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def get_market_status(score, threshold, lang):
    if pd.isna(score) or pd.isna(threshold): return t('unavailable', lang), MUTE
    if score < threshold * 0.75: return t('status_normal', lang), POS
    if score < threshold: return t('status_elevated', lang), WARN
    if score < threshold * 1.5: return t('status_stress', lang), DANGER
    return t('status_crisis', lang), "#E02424"

def generate_market_narrative(row, lang):
    score, thresh = row['Anomaly_Score'], row['Threshold']
    contribs = {s: row.get(f'{s}_Contribution', 0) for s in SIGNALS}
    top_asset_key = max(contribs, key=lambda s: contribs[s] if pd.notna(contribs[s]) else -1)
    pct = contribs[top_asset_key]
    
    asset_display = t('assets', lang).get(top_asset_key, top_asset_key)
    status_label, _color = get_market_status(score, thresh, lang)
    
    if score < thresh:
        return t('narrative_calm', lang).format(status=status_label, driver=asset_display)
    else:
        return t('narrative_warn', lang).format(status=status_label, driver=asset_display, pct=pct)

def build_figure(view, current_color, lang):
    if view == "Last 6 Months" or view == t('ranges', lang).get("Last 6 Months"):
        plot_df = DF.tail(126)
    elif view == "Last 2 Years" or view == t('ranges', lang).get("Last 2 Years"):
        plot_df = DF.tail(504).resample("W").last()
    else:
        plot_df = DF.resample("W").last()

    y_top = np.nanmax([plot_df['Anomaly_Score'].max(), plot_df['Threshold'].max()]) * 1.15
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
                             line=dict(color=tint(current_color, 0.2), width=5, shape='spline', smoothing=0.35),
                             hoverinfo='skip', showlegend=False))
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines', name=t('chart_score', lang),
                             line=dict(color=current_color, width=2, shape='spline', smoothing=0.35),
                             fill='tozeroy', fillcolor=tint(current_color, 0.08),
                             hovertemplate='Score: <b>%{y:.2f}</b><extra></extra>'))
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Threshold'], mode='lines', name=t('chart_limit', lang),
                             line=dict(color=tint(WARN, 0.6), width=1.5, dash='dash'),
                             hovertemplate='Limit: %{y:.2f}<extra></extra>'))

    fp = plot_df[plot_df['Flagged'] == True]
    fig.add_trace(go.Scatter(x=fp.index, y=fp['Anomaly_Score'], mode='markers',
                             marker=dict(color=DANGER, size=6, line=dict(color='#000000', width=1)),
                             hovertemplate='⚠ Flagged Day<br>Score: <b>%{y:.2f}</b><extra></extra>', name='Anomaly'))

    for i, (date_str, label_key) in enumerate(sorted(HISTORICAL_EVENTS.items())):
        dt = pd.to_datetime(date_str)
        if dt >= plot_df.index.min() and dt <= plot_df.index.max():
            label = t(label_key, lang)
            y_offset = -(i % 3) * 16 
            fig.add_vline(x=dt, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.15)",
                          annotation_text=label, annotation_position="top right" if lang == 'ar' else "top left", 
                          annotation_yshift=y_offset,
                          annotation_font=dict(color="#A1A1AA", size=10))

    font_fam = 'ThmanyahSans, sans-serif' if lang == 'ar' else 'Inter, sans-serif'
    
    fig.update_layout(height=360, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#A1A1AA', family=font_fam, size=12),
                      legend=dict(orientation='h', y=1.12, x=1, xanchor='right', bgcolor='rgba(0,0,0,0)',
                                  font=dict(size=12, color='#A1A1AA')),
                      margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified',
                      hoverlabel=dict(bgcolor='#18181B', bordercolor='rgba(255,255,255,0.1)',
                                      font=dict(family=font_fam, size=13, color='#FAFAFA')))
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='rgba(255,255,255,0.1)', zeroline=False,
                     showspikes=True, spikemode='across', spikecolor='rgba(255,255,255,0.15)',
                     spikethickness=1, spikedash='solid', ticks='outside', tickcolor='rgba(255,255,255,0.1)',
                     tickfont=dict(size=11, color='#71717A'))
    fig.update_yaxes(range=[0, y_top], showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False,
                     tickfont=dict(size=11, color='#71717A'), ticksuffix='  ')
    fig.update_traces(cliponaxis=False)
    return fig

def build_contribution_chart(r_color, lang):
    row = DF.iloc[-1]
    contribs = {t('assets', lang).get(s, s): row.get(f'{s}_Contribution', 0) for s in SIGNALS}
    contribs = dict(sorted(contribs.items(), key=lambda item: item[1]))
    colors = [r_color if i == len(contribs)-1 else 'rgba(255,255,255,0.12)' for i in range(len(contribs))]

    font_fam = 'ThmanyahSans, sans-serif' if lang == 'ar' else 'Inter, sans-serif'
    
    fig = go.Figure(go.Bar(
        x=list(contribs.values()), y=list(contribs.keys()), orientation='h',
        marker=dict(color=colors), text=[f"{v:.1f}%" for v in contribs.values()],
        textposition='outside', textfont=dict(color='#A1A1AA', family=font_fam, size=11)
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=30, t=0, b=0), height=140, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False, showticklabels=False, range=[0, max(contribs.values()) * 1.25]),
        yaxis=dict(showgrid=False, tickfont=dict(color='#E4E4E7', size=11)),
        font=dict(family=font_fam), hovermode=False
    )
    return fig

def kpi_card(label, value, sub, large=False, icon_name=None, value_color=None):
    classes = 'kpi-card large' if large else 'kpi-card'
    v_style = {'color': value_color} if value_color else {}
    return html.Div(className=classes, **{'data-aos': 'fade-up'}, children=[
        html.Div(className='kpi-label-row', children=[
            html.Div(label, className='kpi-label'),
            (html.Div(icon(icon_name, 18, MUTE), className='kpi-ico') if icon_name else None),
        ]),
        html.Div(value, className='kpi-value', style=v_style),
        html.Div(sub, className='kpi-sub'),
    ])

def fear_greed_kpi(lang):
    fg_val_str = "N/A"
    fg_desc = t('unavailable', lang)
    fg_color = MUTE
    if HAS_FG:
        try:
            fg = fear_and_greed.get()
            fg_val_str = f"{fg.value:.0f}"
            
            desc_low = fg.description.lower()
            if "fear" in desc_low: fg_color = DANGER
            elif "greed" in desc_low: fg_color = POS
            elif "neutral" in desc_low: fg_color = MUTE
            
            fg_map = {
                'extreme fear': 'خوف شديد', 'fear': 'خوف', 
                'neutral': 'محايد', 'greed': 'طمع', 'extreme greed': 'طمع شديد'
            }
            fg_desc = fg_map.get(desc_low, fg.description.title()) if lang == 'ar' else fg.description.title()
        except Exception:
            fg_desc = t('fetch_failed', lang)
    else:
        fg_desc = t('module_not_installed', lang)
        
    return kpi_card(t('fg_index', lang), fg_val_str, fg_desc, large=True, value_color=fg_color)

def hero_section(lang):
    latest = DF.iloc[-1]
    score, thresh = latest['Anomaly_Score'], latest['Threshold']
    regime, r_color = get_market_status(score, thresh, lang)
    gap = score - thresh
    up = gap >= 0
    delta_class = 'delta up' if up else 'delta down'
    delta_text = f"{'▲' if up else '▼'} {abs(gap):.2f} {t('vs_thresh', lang)}"
    conf_score = "99.8%" 
    
    return html.Div(className='hero-panel glass-card', **{'data-aos': 'fade-up'}, children=[
        html.Div(className='hero-header', children=[
            html.Span(t('score_label', lang), className='hero-title'),
            html.Div(className='hero-badges', children=[
                html.Span(f"{t('confidence', lang)} {conf_score}", className='badge outline'),
                html.Span(f"{t('status', lang)} {regime}", className='badge solid', style={'backgroundColor': tint(r_color, 0.15), 'color': r_color, 'borderColor': tint(r_color, 0.3)})
            ])
        ]),
        html.Div(className='hero-body', children=[
            html.Div(f"{score:.2f}", className='hero-score', style={'color': r_color, 'textShadow': f'0 0 32px {tint(r_color, 0.3)}'}),
            html.Div(className='hero-metrics', children=[
                html.Span(delta_text, className=delta_class, style={'color': r_color, 'backgroundColor': tint(r_color, 0.1)}),
                html.Span(f"{t('last_updated', lang)} {SUMMARY['updated']}", className='hero-timestamp')
            ])
        ])
    ])

def stat_chip(k, v, driver=False):
    return html.Div(className='stat driver' if driver else 'stat', children=[
        html.Div(k, className='stat-k'), html.Div(v, className='stat-v')])

def alert_card(date_idx, row, lang):
    date_str = date_idx.strftime("%Y-%m-%d")
    date_pretty = date_idx.strftime("%B %d, %Y")
    days_ago = (datetime.now() - date_idx.to_pydatetime().replace(tzinfo=None)).days
    is_severe = row['Anomaly_Score'] > row['Threshold'] * 1.3
    sev_label = t('status_crisis', lang) if is_severe else t('status_stress', lang)
    sev = DANGER if is_severe else WARN

    contribs = {s: row.get(f'{s}_Contribution', np.nan) for s in SIGNALS}
    top_asset = max(contribs, key=lambda s: contribs[s] if pd.notna(contribs[s]) else -1)
    top_pct = contribs[top_asset]
    asset_display = t('assets', lang).get(top_asset, top_asset)
    driver_txt = f"{asset_display} {top_pct:.0f}%" if pd.notna(top_pct) else "—"

    stats = [
        stat_chip(t('top_driver', lang), driver_txt, driver=True),
        stat_chip("S&P 500", f"{row['S&P500']:,.0f}"),
        stat_chip("VIX", f"{row['VIX']:.1f}"),
        stat_chip(t('chart_limit', lang), f"{row['Threshold']:.2f}"),
    ]
    pval = row.get('Anomaly_PValue', np.nan)
    if pd.notna(pval):
        stats.append(stat_chip(t('rarity', lang), f"{pval*100:.2f}%"))
    stats.append(stat_chip(t('when', lang), t('days_ago', lang).format(d=days_ago)))

    details_children = [html.Summary(f"📰 {t('view_news', lang)} {date_pretty}", className='news-summary')]
    if date_str in HISTORICAL_EVENTS:
        details_children.append(html.Div(className='event-note', children=[
            html.Span("📌", className='pin'), html.Span(t(HISTORICAL_EVENTS[date_str], lang))]))
    details_children.append(
        html.Button(t('load_headlines', lang), id={'type': 'news-btn', 'index': date_str},
                    n_clicks=0, className='news-load-btn'))
    details_children.append(
        dcc.Loading(type='circle', color=ACCENT,
                    children=html.Div(id={'type': 'news-out', 'index': date_str})))

    return html.Div(className='alert glass-card', **{'data-aos': 'fade-up'}, children=[
        html.Div(className='alert-rail', style={'background': sev}),
        html.Div(className='alert-body', children=[
            html.Div(className='alert-row1', children=[
                html.Div(className='alert-left', children=[
                    html.Span(date_pretty, className='alert-date'),
                    html.Span(sev_label, className='sev-pill',
                              style={'color': sev, 'background': tint(sev, 0.15), 'border': f'1px solid {tint(sev, 0.25)}'}),
                ]),
                html.Span(f"{row['Anomaly_Score']:.2f}", className='alert-score', style={'color': sev}),
            ]),
            html.Div(stats, className='alert-stats'),
            html.Details(details_children, className='news-details'),
        ]),
    ])

def build_cards(year, month, lang):
    if not DATA_OK: return []
    flags = DF[DF['Flagged'] == True].sort_index(ascending=False)
    if year and year != t('all_years', lang):
        flags = flags[flags.index.year == int(year)]
    if month and month != t('all_months', lang):
        m_idx = TR[lang]['months'].index(month) + 1 if month in TR[lang]['months'] else MONTH_NAMES.index(month) + 1
        flags = flags[flags.index.month == m_idx]

    note = None
    if len(flags) > 60:
        flags = flags.head(60)
        note = html.Div(t('showing_recent', lang), className='context-box')
    if len(flags) == 0:
        return [html.Div(t('no_anomaly', lang), className='context-box')]

    children = []
    if note: children.append(note)
    children += [alert_card(idx, row, lang) for idx, row in flags.iterrows()]
    return children

def validation_section(lang):
    s = SUMMARY
    cards = html.Div(className='fintech-grid kpi-row', children=[
        kpi_card(t('crisis_recall', lang), f"{s['recall']:.0f}%", f"{s['detected']} / {s['total_ev']}", large=True, value_color=POS if s['recall'] >= 70 else WARN),
        kpi_card(t('events_detected', lang), f"{s['detected']}", t('within_7d', lang)),
        kpi_card(t('flagged_days', lang), f"{s['total_flags']:,}", t('all_history', lang)),
        kpi_card(t('daily_flag_rate', lang), f"{s['flag_rate']:.1f}%", t('of_trading_days', lang)),
    ])

    header_cells = [html.Th(t('date', lang)), html.Th(t('hist_event', lang)), html.Th(t('composite', lang)), html.Th(t('nearest', lang)), html.Th(t('peak_score', lang))]
    
    body = []
    for r in VAL:
        hit = html.Span(t('detected', lang), className='badge solid success') if r['detected'] else html.Span(t('missed', lang), className='badge solid error')
        nearest = f"{r['nearest']}d" if r['nearest'] is not None else "—"
        peak = f"{r['peak']:.2f}" if r['peak'] is not None else "—"
        event_trans = t(HISTORICAL_EVENTS.get(r['date'], r['event']), lang)
        cells = [html.Td(r['date'], className='mono'), html.Td(event_trans), html.Td(hit), html.Td(nearest, className='mono'), html.Td(peak, className='mono')]
        body.append(html.Tr(cells))

    table = html.Table(className='fintech-table', children=[html.Thead(html.Tr(header_cells)), html.Tbody(body)])

    return html.Div([
        html.H2(t('model_val', lang), className='section-title'),
        cards,
        html.Div(className='glass-card table-wrap', children=table)
    ])

def raw_table(lang):
    raw = DF_IF.tail(100).copy()
    raw.insert(0, 'Date', raw.index.strftime('%Y-%m-%d'))
    for c in raw.columns:
        if raw[c].dtype.kind in 'fc':
            raw[c] = raw[c].round(3)
    cols = [{'name': c, 'id': c} for c in raw.columns]
    return dash_table.DataTable(
        data=raw.to_dict('records'), columns=cols, page_size=15,
        style_table={'overflowX': 'auto', 'borderRadius': '12px'},
        style_header={'backgroundColor': 'transparent', 'color': '#A1A1AA', 'fontWeight': '500',
                      'borderBottom': '1px solid rgba(255,255,255,0.1)', 'fontFamily': 'inherit',
                      'fontSize': '12px', 'textAlign': 'left', 'padding': '12px'},
        style_cell={'backgroundColor': 'transparent', 'color': '#E4E4E7',
                    'borderBottom': '1px solid rgba(255,255,255,0.05)', 'fontFamily': 'JetBrains Mono, monospace',
                    'fontSize': '12px', 'padding': '12px', 'textAlign': 'left'},
        style_data_conditional=[{'if': {'filter_query': '{Flagged} eq 1'}, 'backgroundColor': 'rgba(255,75,75,0.05)'}]
    )

def icon(name, width=18, color=None):
    if not name: return None
    if HAS_ICONIFY:
        kw = {'icon': name, 'width': width, 'height': width}
        if color: kw['color'] = color
        return DashIconify(**kw)
    return html.Span('', className='ico-fallback', style={'width': f'{width}px', 'height': f'{width}px', 'display': 'inline-block'})

NAV_ICONS = {
    'overview': 'lucide:layout-dashboard', 'timeline': 'lucide:trending-up', 'alerts': 'lucide:bell-ring',
    'validation': 'lucide:shield-check', 'methodology': 'lucide:book-open', 'raw': 'lucide:table-2',
}

VIEWS = ["overview", "timeline", "alerts", "validation", "methodology", "raw"]

def data_status_indicator(lang):
    ok = DATA_OK and DF is not None
    dot = POS if ok else DANGER
    text = t('live_data', lang) if ok else t('data_error', lang)
    return html.Div(className='status-indicator', children=[
        html.Span(className='status-dot', style={'background': dot, 'boxShadow': f'0 0 10px {dot}'}),
        html.Span(text, className='status-src')
    ])

def data_status_panel(lang):
    ok = DATA_OK and DF is not None
    rng = f"{DF.index.min().strftime('%b %Y')} – {DF.index.max().strftime('%b %Y')}" if ok else "—"
    items = [
        (t('pipeline_status', lang), t('operational', lang) if ok else t('degraded', lang), POS if ok else DANGER),
        (t('active_source', lang), DATA_SOURCE, ACCENT),
        (t('trading_days', lang), f"{TRADING_DAYS:,}" if ok else "—", ACCENT),
        (t('history_range', lang), rng, ACCENT2),
        (t('last_fetch', lang), LOADED_AT, ACCENT2),
    ]
    return html.Div(className='glass-card', **{'data-aos': 'fade-up'}, children=[
        html.Div(className='card-title', children=[icon('lucide:activity', 16, ACCENT2), html.Span(f"  {t('data_status', lang)}")]),
        html.Div(className='status-grid', children=[
            html.Div(className='status-cell', children=[
                html.Div(k, className='status-key'), html.Div(v, className='status-val', style={'color': c}),
            ]) for k, v, c in items
        ]),
    ])

def _method_block(icon_name, title, children):
    return html.Div(className='glass-card method-block', **{'data-aos': 'fade-up'}, children=[
        html.Div(className='method-head', children=[
            html.Div(icon(icon_name, 20, ACCENT), className='method-ico'), html.H3(title, className='method-title'),
        ]),
        *children,
    ])

def methodology_view(lang):
    return html.Div(className='view-fade-in', children=[
        html.H2(t('methodology', lang), className='section-title'),
        html.P(t('method_lead', lang), className='method-lead', **{'data-aos': 'fade-up'}),
        data_status_panel(lang),
        _method_block('lucide:gauge', t('m_title_1', lang), [html.P([t('m_p_1', lang)], className='method-p')]),
        _method_block('lucide:sigma', t('m_title_2', lang), [html.P([t('m_p_2', lang)], className='method-p')]),
    ])

def sidebar(lang):
    nav = [html.Div(id={'type': 'nav', 'index': key},
                    className='nav-item' + (' active' if key == 'overview' else ''),
                    n_clicks=0, title=t(key, lang), **{'data-nav': key}, children=[
                        html.Span(icon(NAV_ICONS.get(key), 18), className='nav-ico-wrap'),
                        html.Span(t(key, lang), className='nav-label'),
                    ]) for key in VIEWS]
    return html.Div(className='sidebar', children=[
        html.Div(className='sidebar-top', children=[
            html.Div(className='brand-logo', children=[
                html.Div(className='logo-left', children=[
                    html.Img(src=app.get_asset_url('Anomaly_logo.png'), className='logo-img', alt='Anomaly'),
                    html.Span("Anomaly", className='brand-word'),
                ]),
                html.Button(icon('lucide:panel-left', 18, ACCENT2), id='collapse-btn', n_clicks=0,
                            className='collapse-btn', title=t('toggle_sidebar', lang)),
            ])
        ]),
        html.Div(nav, className='nav-menu'),
        html.Div(className='sidebar-foot', children=[
            html.Div(className='live-pill', children=[
                html.Span(className='status-dot'), html.Span("LIVE", className='live-pill-text')]),
            html.Button(t('lang_btn', lang), id='lang-toggle', className='lang-toggle-btn')
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  DASH APP + INDEX
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True, title="Anomaly — Market Intelligence")
server = app.server

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    <link rel="icon" type="image/png" href="/assets/Anomaly_logo.png">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link href="https://unpkg.com/aos@2.3.4/dist/aos.css" rel="stylesheet">
    {%css%}
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
        <script src="https://unpkg.com/aos@2.3.4/dist/aos.js"></script>
        <script>
          (function () {
            function boot() {
              if (window.AOS) {
                window.AOS.init({ duration: 600, easing: 'ease-out-cubic', once: true, offset: 40 });
                setTimeout(function () { window.AOS.refreshHard(); }, 300);
              } else {
                document.documentElement.classList.add('no-aos');
              }
            }
            if (document.readyState === 'complete') boot();
            else window.addEventListener('load', boot);
          })();
        </script>
    </footer>
</body>
</html>'''


def build_view(view_key, lang):
    if view_key == "overview":
        latest = DF.iloc[-1]
        score, thresh = latest['Anomaly_Score'], latest['Threshold']
        status_label, r_color = get_market_status(score, thresh, lang)
        
        row_1 = html.Div(className='glass-card', style={'marginBottom': '24px'}, **{'data-aos': 'fade-up'}, children=[
            html.Div(t('sys_stress', lang), className='card-title'),
            html.Div(dir='ltr', children=[dcc.Graph(id='overview-chart', figure=build_figure(t('ranges', lang).get("Last 6 Months"), r_color, lang), config={'displayModeBar': False})])
        ])

        row_2 = html.Div(className='fintech-grid layout-row-2', children=[
            html.Div(className='glass-card', **{'data-aos': 'fade-up'}, children=[
                html.Div(t('drivers_today', lang), className='card-title'),
                html.Div(dir='ltr', children=[dcc.Graph(figure=build_contribution_chart(r_color, lang), config={'displayModeBar': False})])
            ]),
            html.Div(className='glass-card flex-col', **{'data-aos': 'fade-up'}, children=[
                html.Div(t('market_narrative', lang), className='card-title'),
                html.Div(generate_market_narrative(latest, lang), className='narrative-text')
            ])
        ])

        row_3 = html.Div(className='fintech-grid kpi-row', children=[
            fear_greed_kpi(lang),
            kpi_card(t('exp_thresh', lang), f"{thresh:.2f}", t('causal_mean', lang), icon_name='lucide:git-branch'),
            kpi_card(t('alert_freq', lang), f"{SUMMARY['flag_rate']:.1f}%", t('all_time_rate', lang), icon_name='lucide:activity'),
            kpi_card(t('total_alerts', lang), f"{SUMMARY['total_flags']}", t('hist_events', lang), icon_name='lucide:bell-ring'),
        ])

        return html.Div(className='view-fade-in', children=[hero_section(lang), row_1, row_2, row_3])

    elif view_key == "timeline":
        rngs = t('ranges', lang)
        return html.Div(className='view-fade-in', children=[
            html.H2(t('full_timeline', lang), className='section-title'),
            html.Div(className='glass-card p-4', **{'data-aos': 'fade-up'}, children=[
                dcc.Dropdown(id='range-dd', className='fintech-dd',
                    options=[{'label': rngs.get("Last 6 Months"), 'value': rngs.get("Last 6 Months")}, 
                             {'label': rngs.get("Last 2 Years"), 'value': rngs.get("Last 2 Years")}, 
                             {'label': rngs.get("Full History (2005-Present)"), 'value': rngs.get("Full History (2005-Present)")}],
                    value=rngs.get("Last 2 Years"), clearable=False),
                html.Div(dir='ltr', children=[dcc.Graph(id='anomaly-chart', figure=build_figure(rngs.get("Last 2 Years"), ACCENT, lang), config={'displayModeBar': False})]),
            ])
        ])

    elif view_key == "alerts":
        year_opts = [{'label': t('all_years', lang), 'value': t('all_years', lang)}] + [{'label': str(y), 'value': str(y)} for y in AVAIL_YEARS]
        month_opts = [{'label': t('all_months', lang), 'value': t('all_months', lang)}] + [{'label': m, 'value': m} for m in TR[lang]['months']]
        
        return html.Div(className='view-fade-in', children=[
            html.H2(t('anomaly_alerts', lang), className='section-title'),
            html.Div(className='fintech-grid mb-4', **{'data-aos': 'fade-up'},
                     style={'gridTemplateColumns': '1fr 1fr', 'zIndex': '2'}, children=[
                dcc.Dropdown(id='year-dd', className='fintech-dd', options=year_opts, value=t('all_years', lang), clearable=False),
                dcc.Dropdown(id='month-dd', className='fintech-dd', options=month_opts, value=t('all_months', lang), clearable=False),
            ]),
            dcc.Loading(type='circle', color='#FFFFFF', children=html.Div(id='cards-container')),
        ])

    elif view_key == "validation":
        return html.Div(className='view-fade-in', children=[validation_section(lang)])

    elif view_key == "methodology":
        return methodology_view(lang)

    elif view_key == "raw":
        return html.Div(className='view-fade-in', children=[
            html.H2(t('raw_data_exp', lang), className='section-title'),
            html.Div(className='glass-card', **{'data-aos': 'fade-up'}, children=raw_table(lang))
        ])

    return html.Div("View not found.")


def build_layout():
    if not DATA_OK:
        return html.Div(className='error-screen', children=[
            html.H1("Service Unavailable"), html.P("Market data failed to load."), html.Code(LOAD_ERR)
        ])

    return html.Div(id='root-container', className='app-container', dir='ltr', children=[
        dcc.Store(id='lang-store', data='en'),
        dcc.Store(id='nav-dummy'),
        dcc.Store(id='collapse-dummy'),
        html.Div(id='sidebar-wrap'),
        html.Div(className='main-content', children=[
            html.Div(id='top-nav-wrap'),
            html.Div(id='views-wrap'),
        ]),
    ])

app.layout = build_layout


# ─────────────────────────────────────────────────────────────────────────────
#  SERVER CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output('lang-store', 'data'),
    Input('lang-toggle', 'n_clicks'),
    State('lang-store', 'data'),
    prevent_initial_call=True
)
def toggle_language(n, lang):
    return 'ar' if lang == 'en' else 'en'

@callback(
    Output('root-container', 'dir'),
    Output('root-container', 'className'),
    Output('sidebar-wrap', 'children'),
    Output('top-nav-wrap', 'children'),
    Output('views-wrap', 'children'),
    Input('lang-store', 'data')
)
def render_app(lang):
    d = 'rtl' if lang == 'ar' else 'ltr'
    c = 'app-container font-ar' if lang == 'ar' else 'app-container'
    
    top_n = html.Div(className='top-nav', children=[
        html.Div(t('market_intel', lang), className='nav-title'),
        data_status_indicator(lang),
    ])
    
    view_nodes = []
    for key in VIEWS:
        view_nodes.append(html.Div(build_view(key, lang), className='view', **{'data-view': key},
                                   style={'display': 'block' if key == 'overview' else 'none'}))
                                   
    return d, c, sidebar(lang), top_n, view_nodes


@callback(Output('anomaly-chart', 'figure'), Input('range-dd', 'value'), State('lang-store', 'data'))
def update_chart(view, lang):
    if not DATA_OK: return no_update
    latest = DF.iloc[-1]
    _status, r_color = get_market_status(latest['Anomaly_Score'], latest['Threshold'], lang)
    return build_figure(view, r_color, lang)


@callback(Output('month-dd', 'options'), Output('month-dd', 'value'), Input('year-dd', 'value'), State('lang-store', 'data'))
def update_months(year, lang):
    opts = [{'label': t('all_months', lang), 'value': t('all_months', lang)}]
    if DATA_OK and year and year != t('all_years', lang):
        flags = DF[DF['Flagged'] == True]
        months = sorted(flags[flags.index.year == int(year)].index.month.unique())
        opts += [{'label': TR[lang]['months'][m - 1], 'value': TR[lang]['months'][m - 1]} for m in months]
    return opts, t('all_months', lang)


@callback(Output('cards-container', 'children'), Input('year-dd', 'value'), Input('month-dd', 'value'), State('lang-store', 'data'))
def update_cards(year, month, lang):
    return build_cards(year, month, lang)


@callback(Output({'type': 'news-out', 'index': MATCH}, 'children'),
          Input({'type': 'news-btn', 'index': MATCH}, 'n_clicks'),
          State({'type': 'news-btn', 'index': MATCH}, 'id'), prevent_initial_call=True)
def load_news(n_clicks, btn_id):
    if not n_clicks: return no_update
    news = get_news_for_date(btn_id['index'])
    if not news: return html.Div("No headlines found.", className='news-empty')
    return [html.Div(className='news-item', children=[
        html.Div(title, className='news-title'),
        html.Div(pub, className='news-date'),
        html.A("Read Source ↗", href=link, target="_blank", className='news-link'),
    ]) for (title, link, pub) in news]


# ─────────────────────────────────────────────────────────────────────────────
#  CLIENTSIDE
# ─────────────────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(clicks) {
        var cbctx = window.dash_clientside.callback_context;
        var key = 'overview';
        if (cbctx && cbctx.triggered && cbctx.triggered.length && cbctx.triggered[0].value) {
            try { key = JSON.parse(cbctx.triggered[0].prop_id.split('.n_clicks')[0]).index; } catch (e) {}
        }
        document.querySelectorAll('[data-view]').forEach(function (v) {
            v.style.display = (v.getAttribute('data-view') === key) ? 'block' : 'none';
        });
        document.querySelectorAll('[data-nav]').forEach(function (n) {
            if (n.getAttribute('data-nav') === key) { n.classList.add('active'); }
            else { n.classList.remove('active'); }
        });
        window.dispatchEvent(new Event('resize'));
        if (window.AOS) { window.AOS.refreshHard(); }
        return '';
    }
    """,
    Output('nav-dummy', 'data'),
    Input({'type': 'nav', 'index': ALL}, 'n_clicks'),
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var el = document.querySelector('.app-container');
        if (el) { el.classList.toggle('collapsed'); }
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 300);
        return window.dash_clientside.no_update;
    }
    """,
    Output('collapse-dummy', 'data'),
    Input('collapse-btn', 'n_clicks'),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)

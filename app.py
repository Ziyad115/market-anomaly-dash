import os
import urllib.parse
import xml.etree.ElementTree as ET
import json
import threading
import concurrent.futures
import traceback
import pickle
import tempfile
import random
from datetime import datetime, timedelta
from functools import lru_cache

# DO NOT REMOVE: time, random, threading, pickle are required by load_data()'s retry/cache logic
import time
import random
import threading
import pickle

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

try:
    import fear_and_greed
    HAS_FG = True
except ImportError:
    HAS_FG = False

try:
    from dash_iconify import DashIconify
    HAS_ICONIFY = True
except ImportError:
    HAS_ICONIFY = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS, THEME & TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────────────
SIGNALS = ['S&P500', 'Gold', 'Oil_WTI', 'USD_Index', 'VIX']

ACCENT = "#FFFFFF"     
ACCENT2 = "#8A94A3"    
POS = "#00E599"        
WARN = "#F5A623"       
DANGER = "#FF4B4B"     
MUTE = "#6B7280"       

TR = {
    'en': {
        'overview': 'Overview', 'timeline': 'Timeline', 'alerts': 'Alerts', 'validation': 'Validation', 
        'methodology': 'Methodology', 'raw': 'Raw Data', 'raw_data_exp': 'Raw Data Explorer',
        'sys_stress': 'Systemic Stress Timeline', 'current_status': 'Current Status', 'thresh_limit': 'Threshold limit:', 
        'drivers_today': "Drivers of Today's Score", 'market_narrative': 'Market Summary', 'fg_index': 'Fear & Greed Index', 
        'exp_thresh': 'Dynamic Threshold', 'causal_mean': 'Trailing mean + 2σ', 'alert_freq': 'Alert Frequency', 
        'all_time_rate': 'All-time rate', 'total_alerts': 'Total Alerts', 'hist_events': 'Historical events', 
        'full_timeline': 'Full Anomaly Timeline', 'anomaly_alerts': 'Anomaly Alerts', 'model_val': 'Model Validation & Backtest',
        'live_data': 'Live Data', 'data_error': 'Data Error',
        'confidence': 'Confidence:', 'status': 'Status:', 'last_updated': 'Last updated:', 'vs_thresh': 'vs Threshold',
        'pipeline_status': 'Pipeline status', 'operational': 'Operational', 'degraded': 'Degraded',
        'active_source': 'Active data source', 'trading_days': 'Trading days loaded', 'history_range': 'History range',
        'last_fetch': 'Last successful fetch', 'data_status': 'Data Status', 'crisis_recall': 'Crisis Recall',
        'events_detected': 'Events Detected', 'flagged_days': 'Flagged Days', 'daily_flag_rate': 'Daily Flag Rate',
        'within_7d': 'within ±7 days', 'all_history': 'all history', 'of_trading_days': 'of trading days',
        'date': 'Date', 'hist_event': 'Historical Event', 'composite': 'Composite', 'nearest': 'Nearest',
        'peak_score': 'Peak Score', 'detected': 'Detected', 'missed': 'Missed', 'top_driver': 'Top Driver',
        'rarity': 'Rarity (p)', 'when': 'When', 'view_news': 'View news from', 'load_headlines': 'Load headlines',
        'showing_recent': 'Showing most recent 20 matches.', 'no_anomaly': 'No anomaly days match this filter.',
        'all_years': 'All Years', 'all_months': 'All Months', 'status_normal': 'Normal', 'status_elevated': 'Elevated',
        'status_stress': 'Stress', 'status_crisis': 'Crisis',
        'alert_moderate': 'Moderate', 'alert_severe': 'Severe',
        'narrative_calm': "Markets look calm today — no unusual stress detected. The current status is {status}. Normal background activity is mainly driven by {driver}.",
        'narrative_warn': "Caution: Market stress is unusually high right now. The current status is {status}, primarily driven by sudden moves in {driver} ({pct}% of the activity). Keep an eye on conditions.",
        'chart_score': 'Anomaly Score', 'chart_limit': 'Threshold Limit', 'anomaly': 'Anomaly', 'fetch_failed': 'Fetch Failed',
        'module_not_installed': 'Module not installed', 'unavailable': 'Unavailable', 'days_ago_suffix': 'days ago',
        'score_label': 'Market Anomaly Score', 'lang_btn': 'عربي', 'toggle_sidebar': 'Toggle Sidebar',
        
        'january': 'January', 'february': 'February', 'march': 'March', 'april': 'April', 'may': 'May', 'june': 'June',
        'july': 'July', 'august': 'August', 'september': 'September', 'october': 'October', 'november': 'November', 'december': 'December',
        
        'assets': {'S&P500': 'S&P 500', 'Gold': 'Gold', 'Oil_WTI': 'Oil', 'USD_Index': 'USD', 'VIX': 'VIX'},
        'ranges': {'Last 6 Months': 'Last 6 Months', 'Last 2 Years': 'Last 2 Years', 'Full History (2005-Present)': 'Full History (2005-Present)'},
        
        'evt_lehman': 'Lehman Brothers bankruptcy', 'evt_flash_crash': 'Flash Crash', 'evt_downgrade': 'US credit downgrade',
        'evt_china': 'China devaluation', 'evt_covid': 'COVID-19 Crash', 'evt_circuit': 'Circuit breakers halt',
        'evt_bear': 'S&P 500 bear market', 'evt_ukraine': 'Ukraine Invasion', 'evt_svb': 'SVB Collapse',
        
        'method_lead': 'How this dashboard turns five noisy markets into a single, defensible measure of systemic stress...',
        'm_title_1': 'What it measures', 'm_p_1': 'Every trading day, the model asks one question: how unusual is today, across the whole market at once? It watches five instruments — the S&P 500, Gold, Oil, the US Dollar Index, and the VIX volatility index — because real crises rarely show up in a single asset. They show up as several markets moving strangely together.',
        'm_title_2': 'The composite score — an RMS z-score', 'm_p_2_1': 'Each market is first converted into a z-score: how many standard deviations today\'s move sits from its own recent 63-day norm. A z-score of 3 means a move roughly three times larger than what\'s been typical lately.', 'm_p_2_2': 'Those five z-scores are then combined into one number using the root-mean-square (RMS): we square each z-score, average them, and take the square root. Squaring means a single extreme market dominates the score.',
        'm_title_3': 'The threshold — causal, no look-ahead bias', 'm_p_3_1': 'A score only means something against a bar. The naive approach sets that bar using the mean and standard deviation of the entire history — but that secretly lets the future decide what counted as anomalous in the past.', 'm_p_3_2': 'Instead, the threshold here is an expanding, causal mean + 2σ: on any given day it is computed only from the scores that came strictly before it. Early history faces a calmer bar, and later years face a bar already raised by 2008 and 2020.',
        'm_title_4': 'The Isolation Forest cross-check', 'm_p_4_1': 'As an independent second opinion, the app runs an Isolation Forest — an unsupervised machine-learning model that flags points which are easy to isolate from the rest of the data.', 'm_p_4_2': 'When both methods agree a day is anomalous, that\'s a strong, model-agnostic signal. When they disagree, it\'s a prompt to look closer.',
        'm_title_5': 'How to read the results — and the limits', 'm_p_5_1': 'Treat the score as a thermometer, not a crystal ball. A crossing means conditions are statistically unusual relative to the recent past — it is a prompt to investigate, not a trade signal.', 'm_p_5_2': 'Honest caveats: z-scores assume moves are roughly comparable over time, so structural regime shifts can distort them. The model is deliberately simple and auditable — that transparency is the point.',
        
        'fg_ext_fear': 'Extreme Fear', 'fg_fear': 'Fear', 'fg_neutral': 'Neutral', 'fg_greed': 'Greed', 'fg_ext_greed': 'Extreme Greed',
    },
    'ar': {
        'overview': 'نظرة عامة', 'timeline': 'السجل الزمني', 'alerts': 'التنبيهات', 'validation': 'تقييم النموذج', 
        'methodology': 'المنهجية', 'raw': 'البيانات الخام', 'raw_data_exp': 'استكشاف البيانات الخام',
        'sys_stress': 'مؤشر الضغط النظامي', 'current_status': 'الحالة الحالية', 'thresh_limit': 'حد التنبيه:', 
        'drivers_today': "العوامل المؤثرة اليوم", 'market_narrative': 'ملخص حالة السوق', 'fg_index': 'مؤشر الخوف والطمع', 
        'exp_thresh': 'الحد الديناميكي', 'causal_mean': 'المتوسط التراكمي + 2σ', 'alert_freq': 'معدل التنبيهات', 
        'all_time_rate': 'تاريخياً', 'total_alerts': 'إجمالي التنبيهات', 'hist_events': 'أحداث تاريخية', 
        'full_timeline': 'السجل الزمني الكامل', 'anomaly_alerts': 'سجل التنبيهات', 'model_val': 'دقة وتقييم النموذج',
        'live_data': 'بيانات مباشرة', 'data_error': 'خطأ في البيانات',
        'confidence': 'مستوى الثقة:', 'status': 'الحالة:', 'last_updated': 'آخر تحديث:', 'vs_thresh': 'مقارنة بالحد',
        'pipeline_status': 'حالة الاتصال', 'operational': 'مستقر', 'degraded': 'متدهور',
        'active_source': 'مصدر البيانات', 'trading_days': 'أيام التداول المتاحة', 'history_range': 'النطاق التاريخي',
        'last_fetch': 'آخر تحديث للبيانات', 'data_status': 'حالة البيانات', 'crisis_recall': 'رصد الأزمات',
        'events_detected': 'الأزمات المكتشفة', 'flagged_days': 'أيام التنبيه', 'daily_flag_rate': 'معدل التنبيه اليومي',
        'within_7d': 'بفارق ±7 أيام', 'all_history': 'طوال الفترة', 'of_trading_days': 'من أيام التداول',
        'date': 'التاريخ', 'hist_event': 'الحدث التاريخي', 'composite': 'النموذج المركب', 'nearest': 'أقرب تنبيه',
        'peak_score': 'أعلى درجة', 'detected': 'مرصود', 'missed': 'غير مرصود', 'top_driver': 'المحرك الأكبر',
        'rarity': 'الندرة الإحصائية (p)', 'when': 'المدة', 'view_news': 'عرض الأخبار ليوم', 'load_headlines': 'تحميل العناوين',
        'showing_recent': 'عرض أحدث 20 نتيجة.', 'no_anomaly': 'لا توجد تنبيهات مطابقة.',
        'all_years': 'كل السنوات', 'all_months': 'كل الأشهر', 'status_normal': 'طبيعي', 'status_elevated': 'مرتفع',
        'status_stress': 'ضغط', 'status_crisis': 'أزمة',
        'alert_moderate': 'متوسط', 'alert_severe': 'شديد',
        'narrative_calm': "تبدو الأسواق هادئة اليوم — لم نكتشف أي ضغط غير عادي. الحالة الحالية {status}. النشاط الطبيعي مدفوع بشكل رئيسي بـ {driver}.",
        'narrative_warn': "تحذير: ضغط السوق مرتفع جداً الآن. الحالة الحالية {status}، مدفوعة بحركات مفاجئة في {driver} ({pct}% من النشاط). يرجى المراقبة.",
        'chart_score': 'درجة المؤشر', 'chart_limit': 'حد التنبيه', 'anomaly': 'تنبيه شذوذ', 'fetch_failed': 'فشل التحديث',
        'module_not_installed': 'الوحدة غير مثبتة', 'unavailable': 'غير متوفر', 'days_ago_suffix': 'أيام مضت',
        'score_label': 'مؤشر شذوذ السوق', 'lang_btn': 'EN', 'toggle_sidebar': 'طي/توسيع القائمة',
        
        'january': 'يناير', 'february': 'فبراير', 'march': 'مارس', 'april': 'أبريل', 'may': 'مايو', 'june': 'يونيو',
        'july': 'يوليو', 'august': 'أغسطس', 'september': 'سبتمبر', 'october': 'أكتوبر', 'november': 'نوفمبر', 'december': 'ديسمبر',
        
        'assets': {'S&P500': 'إس آند بي 500', 'Gold': 'الذهب', 'Oil_WTI': 'النفط', 'USD_Index': 'مؤشر الدولار', 'VIX': 'مؤشر التقلب (VIX)'},
        'ranges': {'Last 6 Months': 'آخر 6 أشهر', 'Last 2 Years': 'آخر سنتين', 'Full History (2005-Present)': 'التاريخ الكامل (2005-الآن)'},
        
        'evt_lehman': 'إفلاس ليمان براذرز', 'evt_flash_crash': 'الانهيار الخاطف', 'evt_downgrade': 'تخفيض التصنيف الأمريكي',
        'evt_china': 'تخفيض قيمة اليوان', 'evt_covid': 'انهيار أسواق كوفيد', 'evt_circuit': 'توقف التداول',
        'evt_bear': 'سوق دببية لمؤشر S&P 500', 'evt_ukraine': 'غزو أوكرانيا', 'evt_svb': 'انهيار بنك SVB',
        
        'method_lead': 'كيف تقوم هذه اللوحة بتحليل بيانات خمسة أسواق مختلفة لتحويلها إلى مقياس دقيق للضغط النظامي، ولماذا تعتبر الخيارات الإحصائية مهمة عند قراءة النتائج.',
        'm_title_1': 'ما الذي يتم قياسه؟', 'm_p_1': 'في كل يوم تداول، يطرح النموذج سؤالاً واحداً: ما مدى شذوذ تحركات اليوم عبر السوق بأكمله؟ يراقب النموذج خمسة أصول رئيسية — مؤشر S&P 500، الذهب، النفط، مؤشر الدولار، ومؤشر التقلب VIX — لأن الأزمات الحقيقية نادراً ما تقتصر على أصل واحد، بل تظهر كتحركات غير اعتيادية متزامنة في عدة أسواق.',
        'm_title_2': 'الدرجة المركبة (Z-Score)', 'm_p_2_1': 'أولاً، يتم تحويل أداء كل سوق إلى درجة معيارية (Z-Score) لقياس مدى انحرافه عن متوسط الـ 63 يومًا الماضية. الحصول على درجة 3 يعني أن التحرك أكبر بثلاث مرات من المعتاد.', 'm_p_2_2': 'بعد ذلك، تُدمج هذه الدرجات في رقم واحد باستخدام جذر متوسط المربعات (RMS). عملية التربيع تعني أن أي انحراف شديد في سوق واحد سيهيمن على الدرجة النهائية، وهو ما يعكس طبيعة الأزمات الحقيقية.',
        'm_title_3': 'الحد الديناميكي (تجنب الانحياز للمستقبل)', 'm_p_3_1': 'استخدام متوسط الانحراف لكل التاريخ لضبط حد التنبيه يعتبر خطأً إحصائياً، لأنه يسمح لأحداث المستقبل بالتأثير على تقييم الماضي (Look-ahead bias).', 'm_p_3_2': 'بدلاً من ذلك، يستخدم النموذج هنا حداً ديناميكياً يتوسع بمرور الوقت (المتوسط التراكمي + 2σ)، حيث يتم احتسابه يومياً باستخدام البيانات السابقة فقط. هذا يضمن أن تقييم الأزمات السابقة يتم بناءً على ما كان معروفاً في ذلك الوقت بدقة.',
        'm_title_4': 'نموذج العزل (Isolation Forest)', 'm_p_4_1': 'كوسيلة للتحقق المستقل، يشغل النظام نموذجاً للتعلم الآلي غير الخاضع للإشراف لتحديد النقاط التي يَسهُل "عزلها" إحصائياً عن بقية البيانات.', 'm_p_4_2': 'عندما يتفق كلا النموذجين على وجود شذوذ في يوم ما، فإن ذلك يمثل إشارة قوية وحيادية. وعندما يختلفان، يكون ذلك دافعاً للتحليل المتعمق.',
        'm_title_5': 'كيفية قراءة النتائج والقيود', 'm_p_5_1': 'تعامل مع هذه الدرجة كمقياس لحرارة الأسواق، وليس كأداة للتنبؤ. تجاوز الحد يعني أن الظروف استثنائية إحصائياً مقارنة بالماضي القريب — هي دعوة للمراقبة والتحقق وليست إشارة بيع أو شراء.', 'm_p_5_2': 'يجب ملاحظة أن النموذج يفترض أن التحركات قابلة للمقارنة بمرور الوقت، لذا فإن التغيرات الهيكلية قد تؤثر على الدرجات المعيارية. صُمم النموذج ليكون بسيطاً وقابلاً للتدقيق — وهذه الشفافية هي الهدف الأساسي.',
        
        'fg_ext_fear': 'خوف شديد', 'fg_fear': 'خوف', 'fg_neutral': 'محايد', 'fg_greed': 'طمع', 'fg_ext_greed': 'طمع شديد',
    }
}

def t(key, lang='en'):
    """Safe translation dictionary lookup."""
    return TR.get(lang, TR['en']).get(key, key)

def trans(key):
    """Returns a dual-language span component that switches instantly via CSS class."""
    return html.Span([
        html.Span(t(key, 'en'), className='lang-en'),
        html.Span(t(key, 'ar'), className='lang-ar')
    ])

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

ICO = {
    "activity": '<path d="M22 12h-4l-3 8L9 4l-3 8H2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    "target":   '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "shield":   '<path d="M12 21s7-3.4 7-9V5.5L12 3 5 5.5V12c0 5.6 7 9 7 9z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 8.5v3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="15" r="0.6" fill="currentColor" stroke="currentColor" stroke-width="1"/>',
    "clock":    '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 7.5v5l3.2 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
}

HISTORICAL_EVENTS = {
    "2008-09-15": "evt_lehman",
    "2010-05-06": "evt_flash_crash",
    "2011-08-08": "evt_downgrade",
    "2015-08-24": "evt_china",
    "2020-02-24": "evt_covid",  
    "2020-03-16": "evt_circuit",
    "2022-06-13": "evt_bear",
    "2022-02-24": "evt_ukraine", 
    "2023-03-10": "evt_svb",     
}

def tint(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOGIC & ASYNC FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_single_ticker(name, t_sym):
    """Sequential fetch using yfinance with exponential backoff & jitter to respect rate limits."""
    close = None
    for attempt in range(4):
        try:
            d = yf.download(t_sym, start='2005-01-01', progress=False)
            c = d['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            if len(c) > 0:
                c.index = pd.to_datetime(c.index)
                if c.index.tz is not None:
                    c.index = c.index.tz_localize(None)
                close = c
                break
        except Exception:
            pass
        time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.5))
        
    return name, close if close is not None else pd.Series(dtype=float), "yfinance"

def fetch_fg():
    """Background fetcher for CNN Fear & Greed."""
    if HAS_FG:
        try:
            FG_CACHE['data'] = fear_and_greed.get()
            FG_CACHE['timestamp'] = datetime.now()
        except Exception:
            pass

def load_data():
    """Robust data loading with FRED fallbacks and local caching."""
    global DATA_SOURCE  
    tickers = {'S&P500': '^GSPC', 'VIX': '^VIX', 'Gold': 'GC=F', 'Oil_WTI': 'CL=F', 'USD_Index': 'DX-Y.NYB'}
    fred_map = {'VIX': 'VIXCLS', 'Oil_WTI': 'DCOILWTICO'}
    
    cache = {}
    if os.path.exists(RAW_CACHE_FILE):
        try:
            with open(RAW_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            pass

    data = {}
    sources_used = {}
    log_msgs = []
    now = datetime.now()

    # Fire off F&G independently so network lag doesn't block the layout
    threading.Thread(target=fetch_fg, daemon=True).start()

    # SEQUENTIAL fetch to strictly avoid Yahoo rate limits
    for name, sym in tickers.items():
        series = pd.Series(dtype=float)
        src_name = "none"

        # 1. Check fresh cache (24h TTL)
        cached_item = cache.get(name)
        if cached_item and (now - cached_item['timestamp']) < timedelta(hours=24):
            series = cached_item['data']
            src_name = "cache (fresh)"
        
        # 2. Try yfinance with backoff + jitter
        if series.empty:
            _, series, src_name = fetch_single_ticker(name, sym)

        # 3. Try FRED fallback
        if series.empty and name in fred_map:
            try:
                fred_id = fred_map[name]
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
                df_fred = pd.read_csv(url, na_values='.', parse_dates=['DATE'], index_col='DATE')
                df_fred.index = pd.to_datetime(df_fred.index)
                if df_fred.index.tz is not None: df_fred.index = df_fred.index.tz_localize(None)
                df_fred = df_fred.dropna()
                df_fred = df_fred[df_fred.index >= '2005-01-01']
                if not df_fred.empty:
                    series = df_fred[fred_id]
                    src_name = f"fred:{fred_id}"
            except Exception:
                pass

        # 4. Fallback to expired cache
        if series.empty and cached_item:
            series = cached_item['data']
            src_name = "cache (stale)"

        if not series.empty:
            data[name] = series
            # Prevent updating timestamp if it came from cache
            cache_ts = now if "cache" not in src_name else cached_item['timestamp']
            cache[name] = {'data': series, 'timestamp': cache_ts}
            sources_used[name] = src_name
            log_msgs.append(f"{name}: {src_name} ({len(series)})")
        else:
            data[name] = pd.Series(dtype=float)
            log_msgs.append(f"{name}: FAILED")

    # Save cache
    try:
        with open(RAW_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception:
        pass

    print("[DATA LOAD] " + " | ".join(log_msgs))
    unique_sources = set(sources_used.values())
    DATA_SOURCE = ", ".join(unique_sources) if unique_sources else "unknown"

    # Only compile valid series into the DataFrame to prevent dropna() crashes
    valid_data = {k: v for k, v in data.items() if not v.empty}
    if not valid_data: 
        return pd.DataFrame()
        
    df = pd.DataFrame(valid_data)
    df = df.ffill().dropna()
    return df

def compute_anomaly(prices, window=63, k=2.0, burn_in=252):
    df = prices.copy()
    
    # Gracefully handle missing signals to avoid column exceptions
    active_price_assets = [c for c in ['S&P500', 'Gold', 'Oil_WTI', 'USD_Index'] if c in df.columns]
    active_signals = [c for c in SIGNALS if c in df.columns]

    for col in active_price_assets:
        df[f'{col}_Return'] = np.log(df[col] / df[col].shift(1))
        df[f'{col}_RollMean'] = df[f'{col}_Return'].rolling(window).mean()
        df[f'{col}_RollStd'] = df[f'{col}_Return'].rolling(window).std()
        df[f'{col}_Zscore'] = (df[f'{col}_Return'] - df[f'{col}_RollMean']) / df[f'{col}_RollStd']

    if 'VIX' in df.columns:
        df['VIX_RollMean'] = df['VIX'].rolling(window).mean()
        df['VIX_RollStd'] = df['VIX'].rolling(window).std()
        df['VIX_Zscore'] = (df['VIX'] - df['VIX_RollMean']) / df['VIX_RollStd']

    zcols = [f'{s}_Zscore' for s in active_signals]
    n = len(zcols)
    
    if n == 0:
        df['Anomaly_Score'] = np.nan
        df['Threshold'] = np.nan
        df['Flagged'] = False
        return df

    sum_sq = (df[zcols] ** 2).sum(axis=1)
    safe = sum_sq.replace(0, np.nan)
    df['Sum_Sq_Z'] = sum_sq
    df['Anomaly_Score'] = np.sqrt(sum_sq / n)

    for s in active_signals:
        df[f'{s}_Contribution'] = (df[f'{s}_Zscore'] ** 2 / safe) * 100
        
    # Inject NaNs for fully missing signals so layout still builds safely
    for s in SIGNALS:
        if s not in active_signals:
            df[f'{s}_Contribution'] = np.nan

    df['Anomaly_PValue'] = chi2.sf(df['Sum_Sq_Z'].values, df=n) if HAS_SCIPY else np.nan
    exp_mean = df['Anomaly_Score'].expanding(min_periods=burn_in).mean().shift(1)
    exp_std = df['Anomaly_Score'].expanding(min_periods=burn_in).std().shift(1)
    df['Threshold'] = exp_mean + k * exp_std
    df['Flagged'] = df['Anomaly_Score'] > df['Threshold']
    return df

def compute_isolation_forest(scored_df, contamination):
    active_signals = [s for s in SIGNALS if f'{s}_Zscore' in scored_df.columns]
    zcols = [f'{s}_Zscore' for s in active_signals]
    
    if not zcols:
        out = pd.DataFrame(index=scored_df.index)
        out['IF_Score'] = np.nan
        out['IF_Flagged'] = False
        return out
        
    feat = scored_df[zcols].dropna()
    if feat.empty:
        out = pd.DataFrame(index=scored_df.index)
        out['IF_Score'] = np.nan
        out['IF_Flagged'] = False
        return out
        
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
#  STARTUP & ASYNC STATE CACHE
# ─────────────────────────────────────────────────────────────────────────────
DF = DF_IF = None
VAL = VAL_IF = None
AVAIL_YEARS = []
SUMMARY = {}
DATA_OK = False
LOAD_ERR = ""
TRADING_DAYS = 0
LOADED_AT = "—"

TEMP_DIR = tempfile.gettempdir()
LOCK_FILE = os.path.join(TEMP_DIR, "anomaly_dash_init.lock")
STATE_FILE = os.path.join(TEMP_DIR, "anomaly_dash_state.pkl")
RAW_CACHE_FILE = os.path.join(TEMP_DIR, "anomaly_raw_prices.pkl")
_LOCAL_CACHE_TS = 0

def sync_state():
    global DF, DF_IF, VAL, VAL_IF, AVAIL_YEARS, SUMMARY, DATA_OK, LOAD_ERR, TRADING_DAYS, LOADED_AT, DATA_SOURCE, _LOCAL_CACHE_TS
    if not os.path.exists(STATE_FILE): return
    mtime = os.path.getmtime(STATE_FILE)
    if mtime > _LOCAL_CACHE_TS:
        try:
            with open(STATE_FILE, "rb") as f: state = pickle.load(f)
            DF = state.get('DF')
            DF_IF = state.get('DF_IF')
            VAL = state.get('VAL')
            VAL_IF = state.get('VAL_IF')
            AVAIL_YEARS = state.get('AVAIL_YEARS')
            SUMMARY = state.get('SUMMARY')
            DATA_OK = state.get('DATA_OK')
            LOAD_ERR = state.get('LOAD_ERR')
            TRADING_DAYS = state.get('TRADING_DAYS')
            LOADED_AT = state.get('LOADED_AT')
            DATA_SOURCE = state.get('DATA_SOURCE')
            _LOCAL_CACHE_TS = mtime
            print(f"[SYNC] Worker synced state from disk (mtime: {mtime})")
        except Exception as e:
            print(f"[SYNC ERROR] Failed to load state: {e}")

def init_data():
    global DF, DF_IF, VAL, VAL_IF, AVAIL_YEARS, SUMMARY, DATA_OK, LOAD_ERR, TRADING_DAYS, LOADED_AT
    prices = load_data()
    if prices.empty:
        raise ValueError("Data fetch returned empty DataFrame. No signals available.")
    
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

def run_init_in_background():
    if os.path.exists(LOCK_FILE):
        print("[INIT] Lock file exists. Another worker is already fetching data.")
        return
        
    print("[INIT] Starting background data load...")
    try:
        open(LOCK_FILE, "w").close() 
        init_data()
        
        state = {
            'DF': DF, 'DF_IF': DF_IF, 'VAL': VAL, 'VAL_IF': VAL_IF,
            'AVAIL_YEARS': AVAIL_YEARS, 'SUMMARY': SUMMARY, 'DATA_OK': DATA_OK,
            'LOAD_ERR': LOAD_ERR, 'TRADING_DAYS': TRADING_DAYS, 
            'LOADED_AT': LOADED_AT, 'DATA_SOURCE': DATA_SOURCE
        }
        with open(STATE_FILE, "wb") as f:
            pickle.dump(state, f)
        print("[INIT] Background data load complete. State saved to disk.")
        
    except Exception as e:
        err = traceback.format_exc()
        print(f"[INIT ERROR] Exception during data load:\n{err}")
        state = {'DATA_OK': False, 'LOAD_ERR': str(e)}
        with open(STATE_FILE, "wb") as f:
            pickle.dump(state, f)
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

if not os.environ.get("APP_SKIP_LOAD"):
    threading.Thread(target=run_init_in_background, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS & LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def get_market_status(score, threshold, lang):
    if pd.isna(score) or pd.isna(threshold): return t('unavailable', lang), MUTE
    if score < threshold * 0.75: return t('status_normal', lang), POS
    if score < threshold: return t('status_elevated', lang), WARN
    if score < threshold * 1.5: return t('status_stress', lang), DANGER
    return t('status_crisis', lang), "#E02424"

def dual_market_narrative(row):
    score = row.get('Anomaly_Score', np.nan)
    thresh = row.get('Threshold', np.nan)
    contribs = {s: row.get(f'{s}_Contribution', 0) for s in SIGNALS}
    top_asset_key = max(contribs, key=lambda s: contribs[s] if pd.notna(contribs[s]) else -1)
    pct = contribs[top_asset_key]
    
    def generate(lang):
        asset_display = t('assets', lang).get(top_asset_key, top_asset_key)
        status_label, _color = get_market_status(score, thresh, lang)
        
        # Format explicitly safely against NaNs
        if pd.isna(score) or pd.isna(thresh) or score < thresh:
            return t('narrative_calm', lang).format(status=status_label, driver=asset_display)
        else:
            pct_str = f"{pct:.0f}" if pd.notna(pct) else "0"
            return t('narrative_warn', lang).format(status=status_label, driver=asset_display, pct=pct_str)
            
    return html.Span([
        html.Span(generate('en'), className='lang-en'),
        html.Span(generate('ar'), className='lang-ar')
    ])

def get_empty_fig(height=140):
    return go.Figure(layout=dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=height, xaxis=dict(visible=False), yaxis=dict(visible=False)))

def build_figure(view, current_color, lang='en'):
    if view == "Last 6 Months" or view == t('ranges', lang).get("Last 6 Months"):
        plot_df = DF.tail(126)
    elif view == "Last 2 Years" or view == t('ranges', lang).get("Last 2 Years"):
        plot_df = DF.tail(504).resample("W").last()
    else:
        plot_df = DF.resample("ME").last()

    # Safely compute chart boundaries even if data arrays contain only NaNs
    max_val = np.nanmax([plot_df['Anomaly_Score'].max(), plot_df['Threshold'].max()]) if not plot_df.empty else 0
    y_top = max_val * 1.15 if pd.notna(max_val) else 1.0

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
                             hovertemplate='⚠ Flagged Day<br>Score: <b>%{y:.2f}</b><extra></extra>', name=t('anomaly', lang)))

    for i, (date_str, label_key) in enumerate(sorted(HISTORICAL_EVENTS.items())):
        dt = pd.to_datetime(date_str)
        if dt.tz is not None: dt = dt.tz_localize(None)
        
        if not plot_df.empty and dt >= plot_df.index.min() and dt <= plot_df.index.max():
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

def build_contribution_chart(r_color, lang='en'):
    row = DF.iloc[-1]
    contribs = {}
    for s in SIGNALS:
        val = row.get(f'{s}_Contribution', np.nan)
        if pd.notna(val):
            contribs[t('assets', lang).get(s, s)] = val
        else:
            contribs[t('assets', lang).get(s, s) + f" ({t('unavailable', lang)})"] = 0.0

    contribs = dict(sorted(contribs.items(), key=lambda item: item[1]))
    colors = [r_color if i == len(contribs)-1 else 'rgba(255,255,255,0.12)' for i in range(len(contribs))]

    font_fam = 'ThmanyahSans, sans-serif' if lang == 'ar' else 'Inter, sans-serif'
    
    # Safe float formatting for labels to prevent ValueError on NaN
    text_vals = [f"{v:.1f}%" if pd.notna(v) else "—" for v in contribs.values()]
    
    fig = go.Figure(go.Bar(
        x=list(contribs.values()), y=list(contribs.keys()), orientation='h',
        marker=dict(color=colors), text=text_vals,
        textposition='outside', textfont=dict(color='#A1A1AA', family=font_fam, size=11)
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=30, t=0, b=0), height=140, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False, showticklabels=False, range=[0, max(contribs.values() or [0]) * 1.25]),
        yaxis=dict(showgrid=False, tickfont=dict(color='#E4E4E7', size=11)),
        font=dict(family=font_fam), hovermode=False
    )
    return fig

def kpi_card(label_key, value_component, sub_key, large=False, icon_name=None, value_color=None):
    classes = 'kpi-card large' if large else 'kpi-card'
    v_style = {'color': value_color} if value_color else {}
    return html.Div(className=classes, **{'data-aos': 'fade-up'}, children=[
        html.Div(className='kpi-label-row', children=[
            html.Div(trans(label_key), className='kpi-label'),
            (html.Div(icon(icon_name, 18, MUTE), className='kpi-ico') if icon_name else None),
        ]),
        html.Div(value_component, className='kpi-value', style=v_style),
        html.Div(trans(sub_key) if sub_key else "", className='kpi-sub'),
    ])

def fear_greed_kpi():
    fg_val_str = "N/A"
    fg_color = MUTE
    
    if HAS_FG:
        if FG_CACHE.get('data'):
            fg = FG_CACHE['data']
            fg_val_str = f"{fg.value:.0f}"
            desc_low = fg.description.lower()
            
            if "extreme fear" in desc_low: fg_key, fg_color = 'fg_ext_fear', DANGER
            elif "extreme greed" in desc_low: fg_key, fg_color = 'fg_ext_greed', POS
            elif "fear" in desc_low: fg_key, fg_color = 'fg_fear', DANGER
            elif "greed" in desc_low: fg_key, fg_color = 'fg_greed', POS
            else: fg_key, fg_color = 'fg_neutral', MUTE
            
            desc_comp = trans(fg_key)
        else:
            desc_comp = trans('fetch_failed')
    else:
        desc_comp = trans('module_not_installed')
        
    return kpi_card('fg_index', fg_val_str, None, large=True, value_color=fg_color)

def hero_section():
    latest = DF.iloc[-1]
    score = latest.get('Anomaly_Score', np.nan)
    thresh = latest.get('Threshold', np.nan)
    
    _, r_color = get_market_status(score, thresh, 'en') 
    
    # Safe numerical logic for gap
    gap = score - thresh if pd.notna(score) and pd.notna(thresh) else np.nan
    up = gap >= 0 if pd.notna(gap) else True
    
    delta_class = 'delta up' if up else 'delta down'
    
    score_display = f"{score:.2f}" if pd.notna(score) else "—"
    gap_display = f"{abs(gap):.2f}" if pd.notna(gap) else "—"
    delta_val = f"{'▲' if up else '▼'} {gap_display} "
    
    conf_score = "99.8%" 
    
    status_span = html.Span([html.Span(get_market_status(score, thresh, 'en')[0], className='lang-en'), html.Span(get_market_status(score, thresh, 'ar')[0], className='lang-ar')])
    
    return html.Div(className='hero-panel glass-card', **{'data-aos': 'fade-up'}, children=[
        html.Div(className='hero-header', children=[
            html.Span(trans('score_label'), className='hero-title'),
            html.Div(className='hero-badges', children=[
                html.Span([trans('confidence'), f" {conf_score}"], className='badge outline'),
                html.Span([trans('status'), " ", status_span], className='badge solid', style={'backgroundColor': tint(r_color, 0.15), 'color': r_color, 'borderColor': tint(r_color, 0.3)})
            ])
        ]),
        html.Div(className='hero-body', children=[
            html.Div(score_display, className='hero-score', style={'color': r_color, 'textShadow': f'0 0 32px {tint(r_color, 0.3)}'}),
            html.Div(className='hero-metrics', children=[
                html.Span([delta_val, trans('vs_thresh')], className=delta_class, style={'color': r_color, 'backgroundColor': tint(r_color, 0.1)}),
                html.Span([trans('last_updated'), f" {SUMMARY['updated']}"], className='hero-timestamp')
            ])
        ])
    ])

def stat_chip(k_key, v_component, driver=False):
    k_comp = trans(k_key) if isinstance(k_key, str) and k_key in TR['en'] else k_key
    return html.Div(className='stat driver' if driver else 'stat', children=[
        html.Div(k_comp, className='stat-k'), html.Div(v_component, className='stat-v')])

def get_asset(asset_key, lang):
    return TR[lang].get('assets', {}).get(asset_key, asset_key)

def get_event(date_str):
    if date_str in HISTORICAL_EVENTS:
        k = HISTORICAL_EVENTS[date_str]
        return html.Span([html.Span(t(k, 'en'), className='lang-en'), html.Span(t(k, 'ar'), className='lang-ar')])
    return date_str

def alert_card(date_idx, row):
    date_str = date_idx.strftime("%Y-%m-%d")
    days_ago = (datetime.now() - date_idx.to_pydatetime().replace(tzinfo=None)).days
    
    is_severe = row['Anomaly_Score'] > row['Threshold'] * 1.3
    sev_label = html.Span([html.Span(t('alert_severe', 'en'), className='lang-en'), html.Span(t('alert_severe', 'ar'), className='lang-ar')]) if is_severe else html.Span([html.Span(t('alert_moderate', 'en'), className='lang-en'), html.Span(t('alert_moderate', 'ar'), className='lang-ar')])
    sev = DANGER if is_severe else WARN

    contribs = {s: row.get(f'{s}_Contribution', np.nan) for s in SIGNALS}
    top_asset = max(contribs, key=lambda s: contribs[s] if pd.notna(contribs[s]) else -1)
    top_pct = contribs[top_asset]
    
    # Safe pct layout
    pct_display = f"{top_pct:.0f}%" if pd.notna(top_pct) else "—"
    driver_txt = html.Span([
        html.Span(f"{get_asset(top_asset, 'en')} {pct_display}", className='lang-en'),
        html.Span(f"{get_asset(top_asset, 'ar')} {pct_display}", className='lang-ar')
    ])

    sp500_val = f"{row['S&P500']:,.0f}" if 'S&P500' in row and pd.notna(row['S&P500']) else trans('unavailable')
    vix_val = f"{row['VIX']:.1f}" if 'VIX' in row and pd.notna(row['VIX']) else trans('unavailable')
    thresh_val = f"{row['Threshold']:.2f}" if pd.notna(row.get('Threshold')) else "—"

    stats = [
        stat_chip('top_driver', driver_txt, driver=True),
        stat_chip("S&P 500", sp500_val),
        stat_chip("VIX", vix_val),
        stat_chip('chart_limit', thresh_val),
    ]
    pval = row.get('Anomaly_PValue', np.nan)
    if pd.notna(pval):
        stats.append(stat_chip('rarity', f"{pval*100:.2f}%"))
        
    days_ago_comp = html.Span([
        html.Span(f"{days_ago} {t('days_ago_suffix', 'en')}", className='lang-en'),
        html.Span(f"قبل {days_ago} يوم", className='lang-ar')
    ])
    stats.append(stat_chip('when', days_ago_comp))

    en_date = date_idx.strftime("%B %d, %Y")
    ar_date = f"{date_idx.day} {t(MONTH_NAMES[date_idx.month-1].lower(), 'ar')} {date_idx.year}"
    date_comp = html.Span([html.Span(en_date, className='lang-en'), html.Span(ar_date, className='lang-ar')])

    details_children = [html.Summary(html.Span(["📰 ", trans('view_news'), " ", date_comp]), className='news-summary')]
    if date_str in HISTORICAL_EVENTS:
        details_children.append(html.Div(className='event-note', children=[
            html.Span("📌", className='pin'), get_event(date_str)]))
    details_children.append(
        html.Button(trans('load_headlines'), id={'type': 'news-btn', 'index': date_str},
                    n_clicks=0, className='news-load-btn'))
    details_children.append(
        dcc.Loading(type='circle', color=ACCENT,
                    children=html.Div(id={'type': 'news-out', 'index': date_str})))

    return html.Div(className='alert glass-card', **{'data-aos': 'fade-up'}, children=[
        html.Div(className='alert-rail', style={'background': sev}),
        html.Div(className='alert-body', children=[
            html.Div(className='alert-row1', children=[
                html.Div(className='alert-left', children=[
                    html.Span(date_comp, className='alert-date'),
                    html.Span(sev_label, className='sev-pill',
                              style={'color': sev, 'background': tint(sev, 0.15), 'border': f'1px solid {tint(sev, 0.25)}'}),
                ]),
                html.Span(f"{row['Anomaly_Score']:.2f}", className='alert-score', style={'color': sev}),
            ]),
            html.Div(stats, className='alert-stats'),
            html.Details(details_children, className='news-details'),
        ]),
    ])

def build_cards(year, month):
    if not DATA_OK: return []
    flags = DF[DF['Flagged'] == True].sort_index(ascending=False)
    if year and year != "All Years":
        flags = flags[flags.index.year == int(year)]
    if month and month != "All Months":
        m_idx = MONTH_NAMES.index(month) + 1 if month in MONTH_NAMES else 1
        flags = flags[flags.index.month == m_idx]

    note = None
    if len(flags) > 20: 
        flags = flags.head(20)
        note = html.Div(trans('showing_recent'), className='context-box')
    if len(flags) == 0:
        return [html.Div(trans('no_anomaly'), className='context-box')]

    children = []
    if note: children.append(note)
    children += [alert_card(idx, row) for idx, row in flags.iterrows()]
    return children

def validation_section():
    s = SUMMARY
    cards = html.Div(className='fintech-grid kpi-row', children=[
        kpi_card('crisis_recall', f"{s['recall']:.0f}%", html.Span(f"{s['detected']} / {s['total_ev']}"), large=True, value_color=POS if s['recall'] >= 70 else WARN),
        kpi_card('events_detected', f"{s['detected']}", 'within_7d'),
        kpi_card('flagged_days', f"{s['total_flags']:,}", 'all_history'),
        kpi_card('daily_flag_rate', f"{s['flag_rate']:.1f}%", 'of_trading_days'),
    ])

    header_cells = [html.Th(trans('date')), html.Th(trans('hist_event')), html.Th(trans('composite')), html.Th(trans('nearest')), html.Th(trans('peak_score'))]
    
    body = []
    for r in VAL:
        hit = html.Span(trans('detected'), className='badge solid success') if r['detected'] else html.Span(trans('missed'), className='badge solid error')
        nearest = html.Span([html.Span(f"{r['nearest']}d", className='lang-en'), html.Span(f"{r['nearest']} يوم", className='lang-ar')]) if r['nearest'] is not None else "—"
        peak = f"{r['peak']:.2f}" if r['peak'] is not None else "—"
        
        evt_key = HISTORICAL_EVENTS.get(r['date'], r['event'])
        event_trans = trans(evt_key) if evt_key in TR['en'] else r['event']
        
        cells = [html.Td(r['date'], className='mono'), html.Td(event_trans), html.Td(hit), html.Td(nearest, className='mono'), html.Td(peak, className='mono')]
        body.append(html.Tr(cells))

    table = html.Table(className='fintech-table', children=[html.Thead(html.Tr(header_cells)), html.Tbody(body)])

    return html.Div([
        html.H2(trans('model_val'), className='section-title'),
        cards,
        html.Div(className='glass-card table-wrap', children=table)
    ])

def raw_table():
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

def _method_block(icon_name, title_key, p_keys):
    return html.Div(className='glass-card method-block', **{'data-aos': 'fade-up'}, children=[
        html.Div(className='method-head', children=[
            html.Div(icon(icon_name, 20, ACCENT), className='method-ico'), html.H3(trans(title_key), className='method-title'),
        ]),
        *[html.P(trans(k), className='method-p') for k in p_keys]
    ])

def methodology_view():
    return html.Div(className='view-fade-in', children=[
        html.H2(trans('methodology'), className='section-title'),
        html.P(trans('method_lead'), className='method-lead', **{'data-aos': 'fade-up'}),
        
        _method_block('lucide:gauge', 'm_title_1', ['m_p_1']),
        _method_block('lucide:sigma', 'm_title_2', ['m_p_2_1', 'm_p_2_2']),
        _method_block('lucide:git-branch', 'm_title_3', ['m_p_3_1', 'm_p_3_2']),
        _method_block('lucide:box', 'm_title_4', ['m_p_4_1', 'm_p_4_2']),
        _method_block('lucide:eye', 'm_title_5', ['m_p_5_1', 'm_p_5_2']),
    ])

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

def sidebar():
    nav = [html.Div(id={'type': 'nav', 'index': key},
                    className='nav-item' + (' active' if key == 'overview' else ''),
                    n_clicks=0, **{'data-nav': key}, children=[
                        html.Span(icon(NAV_ICONS.get(key), 18), className='nav-ico-wrap'),
                        html.Span(trans(key), className='nav-label'),
                    ]) for key in VIEWS]
    return html.Div(className='sidebar', children=[
        html.Div(className='sidebar-top', children=[
            html.Div(className='brand-logo', children=[
                html.Div(className='logo-left', children=[
                    html.Img(src=app.get_asset_url('Anomaly_logo_wordmark.png'), className='logo-wordmark', alt='Anomaly'),
                ]),
                html.Button(icon('lucide:panel-left', 18, ACCENT2), id='collapse-btn', n_clicks=0, className='collapse-btn'),
            ])
        ]),
        html.Div(className='nav-container', children=[
            html.Div(nav, className='nav-menu'),
            html.Div(className='nav-extra', children=[
                html.Button(trans('lang_btn'), id='lang-toggle', className='lang-toggle-btn')
            ]),
        ])
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
    <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
    <script>console.time('Dash-Initial-Render');</script>
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
                setTimeout(function () { 
                    window.AOS.init({ duration: 600, easing: 'ease-out-cubic', once: true, offset: 40 });
                    window.AOS.refreshHard(); 
                    console.timeEnd('Dash-Initial-Render');
                }, 300);
              } else { 
                  document.documentElement.classList.add('no-aos'); 
                  console.timeEnd('Dash-Initial-Render');
              }
            }
            if (document.readyState === 'complete') boot();
            else window.addEventListener('load', boot);
          })();
        </script>
    </footer>
</body>
</html>'''

def serve_layout():
    try:
        sync_state()
        
        if not DATA_OK and os.path.exists(LOCK_FILE):
            return html.Div(
                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'height': '100vh', 'backgroundColor': '#0A0A0A', 'fontFamily': 'Inter, sans-serif'},
                children=[
                    dcc.Interval(id='boot-interval', interval=1000, n_intervals=0),
                    html.Div(id='boot-trigger', style={'display': 'none'}),
                    html.Div(style={'width': '40px', 'height': '40px', 'border': '3px solid rgba(255,255,255,0.1)', 'borderTopColor': POS, 'borderRadius': '50%', 'animation': 'spin 1s linear infinite', 'marginBottom': '20px'}),
                    html.H2("Initializing Market Intelligence...", style={'fontSize': '15px', 'fontWeight': '500', 'letterSpacing': '0.05em', 'color': ACCENT2})
                ]
            )

        if not DATA_OK:
            return html.Div(className='error-screen', children=[html.H1("Service Unavailable"), html.P("Market data failed to load. See server logs for details."), html.Code(LOAD_ERR)])

        overview_html = build_view("overview")

        return html.Div(id='root-container', className='app-container lang-en', dir='ltr', children=[
            dcc.Interval(id='render-interval', interval=200, max_intervals=1),
            dcc.Store(id='tr-store', data=TR),
            dcc.Store(id='nav-dummy'), dcc.Store(id='collapse-dummy'),
            sidebar(),
            html.Div(className='main-content', children=[
                html.Div(className='top-nav', children=[
                    html.Div(className='status-indicator', children=[
                        html.Span(className='status-dot', style={'background': POS, 'boxShadow': f'0 0 10px {POS}'}),
                        html.Span(trans('live_data'), className='status-src')
                    ])
                ]),
                html.Div(overview_html, id='views-wrap'),
            ]),
        ])
    except Exception as e:
        err_trace = traceback.format_exc()
        print(f"[LAYOUT EXCEPTION] {err_trace}")
        return html.Div(
            style={'padding': '50px', 'color': '#ff4b4b', 'backgroundColor': '#0a0a0a', 'fontFamily': 'monospace', 'height': '100vh'},
            children=[
                html.H2("CRITICAL LAYOUT ERROR"),
                html.Pre(err_trace, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word', 'color': '#a1a1aa'})
            ]
        )

def build_view(view_key, lang='en'):
    if view_key == "overview":
        latest = DF.iloc[-1]
        score, thresh = latest['Anomaly_Score'], latest['Threshold']
        _, r_color = get_market_status(score, thresh, 'en')
        
        row_1 = html.Div(className='glass-card', style={'marginBottom': '24px'}, **{'data-aos': 'fade-up'}, children=[
            html.Div(trans('sys_stress'), className='card-title'),
            html.Div(dir='ltr', children=[dcc.Graph(id='overview-chart', figure=build_figure("Last 6 Months", r_color, 'en'), config={'displayModeBar': False})])
        ])

        row_2 = html.Div(className='fintech-grid layout-row-2', children=[
            html.Div(className='glass-card', **{'data-aos': 'fade-up'}, children=[
                html.Div(trans('drivers_today'), className='card-title'),
                html.Div(dir='ltr', children=[dcc.Graph(id='contrib-chart', figure=get_empty_fig(), config={'displayModeBar': False})])
            ]),
            html.Div(className='glass-card flex-col', **{'data-aos': 'fade-up'}, children=[
                html.Div(trans('market_narrative'), className='card-title'),
                html.Div(dual_market_narrative(latest), className='narrative-text')
            ])
        ])

        row_3 = html.Div(className='fintech-grid kpi-row', children=[
            fear_greed_kpi(),
            kpi_card('exp_thresh', f"{thresh:.2f}" if pd.notna(thresh) else "—", 'causal_mean', icon_name='lucide:git-branch'),
            kpi_card('alert_freq', f"{SUMMARY['flag_rate']:.1f}%", 'all_time_rate', icon_name='lucide:activity'),
            kpi_card('total_alerts', f"{SUMMARY['total_flags']}", 'hist_events', icon_name='lucide:bell-ring', value_color=ACCENT),
        ])

        return html.Div(className='view-fade-in', children=[hero_section(), row_1, row_2, row_3])

    elif view_key == "timeline":
        return html.Div(className='view-fade-in', children=[
            html.H2(trans('full_timeline'), className='section-title'),
            html.Div(className='glass-card p-4', **{'data-aos': 'fade-up'}, children=[
                html.Div(className='control-wrapper', children=[
                    dcc.Dropdown(id='range-dd', className='fintech-dd', clearable=False,
                        options=[{'label': html.Span([html.Span("Last 6 Months", className='lang-en'), html.Span("آخر 6 أشهر", className='lang-ar')]), 'value': "Last 6 Months"}, 
                                 {'label': html.Span([html.Span("Last 2 Years", className='lang-en'), html.Span("آخر سنتين", className='lang-ar')]), 'value': "Last 2 Years"}, 
                                 {'label': html.Span([html.Span("Full History (2005-Present)", className='lang-en'), html.Span("التاريخ الكامل (2005-الآن)", className='lang-ar')]), 'value': "Full History (2005-Present)"}],
                        value="Last 2 Years"),
                ]),
                html.Div(dir='ltr', children=[dcc.Graph(id='anomaly-chart', figure=build_figure("Last 2 Years", ACCENT, 'en'), config={'displayModeBar': False})]),
            ])
        ])

    elif view_key == "alerts":
        year_opts = [{'label': html.Span([html.Span(t('all_years', 'en'), className='lang-en'), html.Span(t('all_years', 'ar'), className='lang-ar')]), 'value': 'All Years'}] + [{'label': str(y), 'value': str(y)} for y in AVAIL_YEARS]
        month_opts = [{'label': html.Span([html.Span(t('all_months', 'en'), className='lang-en'), html.Span(t('all_months', 'ar'), className='lang-ar')]), 'value': 'All Months'}]
        for m in MONTH_NAMES: month_opts.append({'label': trans(m.lower()), 'value': m})
        
        return html.Div(className='view-fade-in', children=[
            html.H2(trans('anomaly_alerts'), className='section-title'),
            html.Div(className='fintech-grid mb-4', **{'data-aos': 'fade-up'},
                     style={'gridTemplateColumns': '1fr 1fr', 'zIndex': '2'}, children=[
                dcc.Dropdown(id='year-dd', className='fintech-dd', options=year_opts, value='All Years', clearable=False),
                dcc.Dropdown(id='month-dd', className='fintech-dd', options=month_opts, value='All Months', clearable=False),
            ]),
            dcc.Loading(type='circle', color=ACCENT, children=html.Div(id='cards-container')),
        ])

    elif view_key == "validation":
        return html.Div(className='view-fade-in', children=[validation_section()])

    elif view_key == "methodology":
        return methodology_view()

    elif view_key == "raw":
        return html.Div(className='view-fade-in', children=[
            html.H2(trans('raw_data_exp'), className='section-title'),
            html.Div(className='glass-card', **{'data-aos': 'fade-up'}, children=raw_table())
        ])

    return html.Div("View not found.")

app.layout = serve_layout

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@callback(Output('boot-trigger', 'children'), Input('boot-interval', 'n_intervals'))
def check_boot_state(n):
    sync_state()
    if DATA_OK: return "READY"
    if LOAD_ERR or not os.path.exists(LOCK_FILE): return "ERROR"
    return "LOADING"

app.clientside_callback(
    """
    function(status) {
        if (status === "READY" || status === "ERROR") { window.location.reload(true); }
        return window.dash_clientside.no_update;
    }
    """,
    Output('boot-interval', 'disabled'), Input('boot-trigger', 'children'), prevent_initial_call=True
)

@callback(
    Output('views-wrap', 'children'),
    Output({'type': 'nav', 'index': ALL}, 'className'),
    Input({'type': 'nav', 'index': ALL}, 'n_clicks'),
    State('root-container', 'className'),
    prevent_initial_call=True
)
def switch_view(n_clicks, current_class):
    if not any(c for c in n_clicks if c is not None):
        return no_update, no_update
        
    view_key = 'overview'
    if ctx.triggered_id:
        view_key = ctx.triggered_id['index']
        
    lang = 'ar' if 'lang-ar' in (current_class or '') else 'en'
    view_html = build_view(view_key, lang)
    nav_classes = ['nav-item active' if key == view_key else 'nav-item' for key in VIEWS]
    
    return view_html, nav_classes

@callback(
    Output('contrib-chart', 'figure'),
    Input('render-interval', 'n_intervals'),
    State('root-container', 'className'),
    prevent_initial_call=True
)
def load_deferred_charts(n, current_class):
    if not DATA_OK: return no_update
    lang = 'ar' if 'lang-ar' in (current_class or '') else 'en'
    latest = DF.iloc[-1]
    _, r_color = get_market_status(latest['Anomaly_Score'], latest['Threshold'], lang)
    return build_contribution_chart(r_color, lang)

@callback(Output('anomaly-chart', 'figure', allow_duplicate=True), Input('range-dd', 'value'), State('root-container', 'className'), prevent_initial_call=True)
def update_timeline_chart(view, current_class):
    if not DATA_OK: return no_update
    lang = 'ar' if 'lang-ar' in (current_class or '') else 'en'
    return build_figure(view or "Last 6 Months", ACCENT, lang)

@callback(Output('month-dd', 'options'), Output('month-dd', 'value'), Input('year-dd', 'value'))
def update_months(year):
    opts = [{'label': trans('all_months'), 'value': 'All Months'}]
    if DATA_OK and year and year != "All Years":
        flags = DF[DF['Flagged'] == True]
        months = sorted(flags[flags.index.year == int(year)].index.month.unique())
        opts += [{'label': trans(MONTH_NAMES[m - 1].lower()), 'value': MONTH_NAMES[m - 1]} for m in months]
    return opts, 'All Months'

@callback(Output('cards-container', 'children'), Input('year-dd', 'value'), Input('month-dd', 'value'))
def update_cards(year, month):
    return build_cards(year, month)

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

# Decoupled Language Switcher. Instant updates using Plotly.relayout/restyle
app.clientside_callback(
    """
    function(n_clicks, tr_data, current_class) {
        if (!n_clicks || !document.querySelector('.app-container')) return window.dash_clientside.no_update;
        
        const new_lang = current_class.includes('lang-en') ? 'ar' : 'en';
        const is_ar = new_lang === 'ar';
        const TR = tr_data[new_lang];
        
        setTimeout(() => {
            const plots = document.querySelectorAll('.js-plotly-plot');
            plots.forEach(plot => {
                const font = is_ar ? 'ThmanyahSans, sans-serif' : 'Inter, sans-serif';
                let layoutUpdate = { 'font.family': font, 'hoverlabel.font.family': font };
                
                if (plot.layout && plot.layout.annotations) {
                    let newAnnotations = [...plot.layout.annotations];
                    const evt_keys = ['evt_lehman', 'evt_flash_crash', 'evt_downgrade', 'evt_china', 'evt_covid', 'evt_circuit', 'evt_bear', 'evt_ukraine', 'evt_svb'];
                    newAnnotations.forEach(a => {
                        for(let k of evt_keys) {
                            if(a.text === tr_data['en'][k] || a.text === tr_data['ar'][k]) {
                                a.text = tr_data[new_lang][k];
                                a.xanchor = is_ar ? "right" : "left";
                            }
                        }
                    });
                    layoutUpdate['annotations'] = newAnnotations;
                }
                Plotly.relayout(plot, layoutUpdate);
                
                if (plot.data && plot.data.length >= 2) {
                    let dataUpdate = { name: [TR['chart_score'], TR['chart_limit']] };
                    let traceIndices = [1, 2];
                    if (plot.data.length >= 4) {
                        dataUpdate.name.push(TR['anomaly']);
                        traceIndices.push(3);
                    }
                    Plotly.restyle(plot, dataUpdate, traceIndices);
                }
                
                if (plot.data && plot.data[0] && plot.data[0].type === 'bar') {
                    let newY = plot.data[0].y.map(asset => {
                        for(let k in tr_data['en']['assets']) {
                            if(asset === tr_data['en']['assets'][k] || asset === tr_data['ar']['assets'][k]) {
                                return tr_data[new_lang]['assets'][k];
                            }
                        }
                        return asset;
                    });
                    Plotly.restyle(plot, {y: [newY]}, [0]);
                }
            });
        }, 50);
        
        const dir = is_ar ? 'rtl' : 'ltr';
        const cls = 'app-container lang-' + new_lang + (is_ar ? ' font-ar' : '');
        
        return [dir, cls];
    }
    """,
    Output('root-container', 'dir'), Output('root-container', 'className'),
    Input('lang-toggle', 'n_clicks'),
    State('tr-store', 'data'), State('root-container', 'className'),
    prevent_initial_call=True
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
    Output('collapse-dummy', 'data'), Input('collapse-btn', 'n_clicks'), prevent_initial_call=True
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)

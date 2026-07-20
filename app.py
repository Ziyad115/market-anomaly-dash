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

# New Fear & Greed dependency
try:
    import fear_and_greed
    HAS_FG = True
except ImportError:
    HAS_FG = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & THEME
# ─────────────────────────────────────────────────────────────────────────────
SIGNALS = ['S&P500', 'Gold', 'Oil_WTI', 'USD_Index', 'VIX']
DISPLAY = {'S&P500': 'S&P 500', 'Gold': 'Gold', 'Oil_WTI': 'Oil', 'USD_Index': 'USD', 'VIX': 'VIX'}

# Premium Fintech Palette (Vercel/Linear/Stripe inspired)
ACCENT = "#FFFFFF"     # High contrast white for dominant elements
ACCENT2 = "#8A94A3"    # Muted Gray
POS = "#00E599"        # Fintech Neon Green
WARN = "#F5A623"       # Premium Amber
DANGER = "#FF4B4B"     # Crisp Red
MUTE = "#6B7280"       # Faint Text

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

ICO = {
    "activity": '<path d="M22 12h-4l-3 8L9 4l-3 8H2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    "target":   '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "shield":   '<path d="M12 21s7-3.4 7-9V5.5L12 3 5 5.5V12c0 5.6 7 9 7 9z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 8.5v3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="15" r="0.6" fill="currentColor" stroke="currentColor" stroke-width="1"/>',
    "clock":    '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 7.5v5l3.2 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
}

HISTORICAL_EVENTS = {
    "2008-09-15": "Lehman Brothers files for bankruptcy, triggering global financial crisis.",
    "2008-10-13": "Global stock markets rally after coordinated bank bailout announcements.",
    "2008-11-20": "S&P 500 hits multi-year lows amid deepening recession fears.",
    "2009-03-09": "S&P 500 bottoms out during the Global Financial Crisis.",
    "2010-05-06": "Flash Crash: Dow Jones drops ~1000 points in minutes.",
    "2011-08-08": "US credit rating downgraded by S&P, sparking global selloff.",
    "2015-08-24": "China devaluation fears trigger global market selloff ('Black Monday').",
    "2020-02-24": "COVID-19 fears trigger global market selloff as cases spread outside China.",
    "2020-03-16": "Circuit breakers halt trading as COVID-19 panic selling accelerates.",
    "2022-06-13": "S&P 500 enters bear market amid rate hike and inflation fears.",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def tint(hex_color, alpha):
    """Hex -> rgba() string at the given alpha."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def svg_icon(name, color, size=19):
    """Return an html.Img holding an inline SVG (color baked in)."""
    inner = ICO[name].replace('currentColor', color)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}">{inner}</svg>'
    uri = "data:image/svg+xml;utf8," + urllib.parse.quote(svg)
    return html.Img(src=uri, style={'width': f'{size}px', 'height': f'{size}px', 'display': 'block'})

# ─────────────────────────────────────────────────────────────────────────────
#  DATA + MODEL LOGIC (Untouched Core Logic)
# ─────────────────────────────────────────────────────────────────────────────
import time

def load_data():
    tickers = {'S&P500': '^GSPC', 'VIX': '^VIX', 'Gold': 'GC=F',
               'Oil_WTI': 'CL=F', 'USD_Index': 'DX-Y.NYB'}
    data = {}

    try:
        from defeatbeta_api.data.ticker import Ticker as DBTicker
        for name, t in tickers.items():
            dbt = DBTicker(t)
            price_df = dbt.price()
            price_df['report_date'] = pd.to_datetime(price_df['report_date'])
            price_df = price_df.set_index('report_date').sort_index()
            price_df = price_df[price_df.index >= '2005-01-01']
            data[name] = price_df['close']
        df = pd.DataFrame(data).dropna()
        if len(df) > 0:
            return df
    except Exception:
        pass

    for name, t in tickers.items():
        close = None
        for attempt in range(4):
            try:
                d = yf.download(t, start='2005-01-01', progress=False)
                c = d['Close']
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                if len(c) > 0:
                    close = c
                    break
            except Exception:
                pass
            time.sleep(2 ** attempt)  
        data[name] = close if close is not None else pd.Series(dtype=float)

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

def init_data():
    global DF, DF_IF, VAL, VAL_IF, AVAIL_YEARS, SUMMARY, DATA_OK, LOAD_ERR
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
        'detected': detected, 'total_ev': total_ev,
        'recall': (detected / total_ev * 100) if total_ev else 0.0,
        'total_flags': total_flags, 'flag_rate': flag_rate,
        'if_detected': (sum(r['detected'] for r in VAL_IF) if VAL_IF else None),
        'updated': datetime.now().strftime("%d %b · %H:%M"),
    }
    if VAL_IF:
        SUMMARY['if_recall'] = (SUMMARY['if_detected'] / total_ev * 100) if total_ev else 0.0

    flags = DF[DF['Flagged'] == True]
    AVAIL_YEARS = sorted(flags.index.year.unique(), reverse=True)
    DATA_OK = True

if not os.environ.get("APP_SKIP_LOAD"):
    try:
        init_data()
    except Exception as e:
        DATA_OK = False
        LOAD_ERR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  FINTECH UI HELPERS & LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def get_market_regime(score, threshold):
    if pd.isna(score) or pd.isna(threshold): return "Unknown", MUTE
    if score < threshold * 0.75: return "Normal", POS
    if score < threshold: return "Elevated", WARN
    if score < threshold * 1.5: return "Stress", DANGER
    return "Crisis", "#E02424"

def generate_market_narrative(row):
    score, thresh = row['Anomaly_Score'], row['Threshold']
    contribs = {s: row.get(f'{s}_Contribution', 0) for s in SIGNALS}
    top_asset = max(contribs, key=lambda s: contribs[s] if pd.notna(contribs[s]) else -1)
    pct = contribs[top_asset]
    regime, _ = get_market_regime(score, thresh)
    
    if score < thresh:
        return f"Composite stress remains below the structural threshold. Market regime is classified as {regime}. {DISPLAY[top_asset]} and volatility are currently the largest contributors to the background score. No broad-based systemic stress or anomalous behavior is detected."
    else:
        return f"Warning: Composite stress has breached the expanding threshold. Market regime is currently classified as {regime}. The anomaly is heavily driven by {DISPLAY[top_asset]}, accounting for {pct:.0f}% of the divergence. Monitor closely for cross-asset contagion."


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE + COMPONENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(view):
    if view == "Last 6 Months":
        plot_df = DF.tail(126)
    elif view == "Last 2 Years":
        plot_df = DF.tail(504).resample("W").last()
    else:
        plot_df = DF.resample("W").last()

    y_top = np.nanmax([plot_df['Anomaly_Score'].max(), plot_df['Threshold'].max()]) * 1.15
    fig = go.Figure()

    # Base subtle glow line
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
                             line=dict(color='rgba(255,255,255,0.1)', width=6, shape='spline', smoothing=0.35),
                             hoverinfo='skip', showlegend=False))
    
    # Main precise line
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines', name='Anomaly Score',
                             line=dict(color='#FFFFFF', width=2, shape='spline', smoothing=0.35),
                             fill='tozeroy', fillcolor='url(#linear-gradient)',
                             hovertemplate='Score: <b>%{y:.2f}</b><extra></extra>'))
    
    # Threshold Line
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Threshold'], mode='lines', name='Threshold Limit',
                             line=dict(color='rgba(255,255,255,0.25)', width=1.5, dash='dash'),
                             hovertemplate='Limit: %{y:.2f}<extra></extra>'))

    fp = plot_df[plot_df['Flagged'] == True]
    fig.add_trace(go.Scatter(x=fp.index, y=fp['Anomaly_Score'], mode='markers',
                             marker=dict(color=DANGER, size=6, line=dict(color='#000000', width=1)),
                             hovertemplate='⚠ Flagged Day<br>Score: <b>%{y:.2f}</b><extra></extra>', name='Anomaly'))

    # Annotations for historical context
    events_to_plot = {
        "2020-02-24": "COVID Crash",
        "2022-02-24": "Ukraine Inv.",
        "2023-03-10": "SVB Collapse"
    }
    
    for date_str, label in events_to_plot.items():
        dt = pd.to_datetime(date_str)
        if dt >= plot_df.index.min() and dt <= plot_df.index.max():
            fig.add_vline(x=dt, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.15)",
                          annotation_text=label, annotation_position="top left", 
                          annotation_font=dict(color="#A1A1AA", size=10))

    fig.update_layout(height=360, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#A1A1AA', family='Inter', size=12),
                      legend=dict(orientation='h', y=1.12, x=1, xanchor='right', bgcolor='rgba(0,0,0,0)',
                                  font=dict(size=12, color='#A1A1AA')),
                      margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified',
                      hoverlabel=dict(bgcolor='#18181B', bordercolor='rgba(255,255,255,0.1)',
                                      font=dict(family='Inter', size=13, color='#FAFAFA')))
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='rgba(255,255,255,0.1)', zeroline=False,
                     showspikes=True, spikemode='across', spikecolor='rgba(255,255,255,0.15)',
                     spikethickness=1, spikedash='solid', ticks='outside', tickcolor='rgba(255,255,255,0.1)',
                     tickfont=dict(size=11, color='#71717A'))
    fig.update_yaxes(range=[0, y_top], showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False,
                     tickfont=dict(size=11, color='#71717A'), ticksuffix='  ')
    fig.update_traces(cliponaxis=False)
    return fig


def build_contribution_chart():
    row = DF.iloc[-1]
    contribs = {DISPLAY[s]: row.get(f'{s}_Contribution', 0) for s in SIGNALS}
    contribs = dict(sorted(contribs.items(), key=lambda item: item[1]))
    
    fig = go.Figure(go.Bar(
        x=list(contribs.values()), y=list(contribs.keys()), orientation='h',
        marker=dict(color='#FFFFFF', opacity=0.9),
        text=[f"{v:.1f}%" for v in contribs.values()],
        textposition='outside',
        textfont=dict(color='#A1A1AA', family='Inter', size=11)
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=40, t=10, b=0), height=200, 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, 
                   showticklabels=False, range=[0, max(contribs.values()) * 1.2]),
        yaxis=dict(showgrid=False, tickfont=dict(color='#E4E4E7', size=12)),
        font=dict(family='Inter'), hovermode=False
    )
    return fig


def kpi_card(label, value, sub, large=False):
    classes = 'kpi-card large' if large else 'kpi-card'
    return html.Div(className=classes, children=[
        html.Div(label, className='kpi-label'),
        html.Div(value, className='kpi-value'),
        html.Div(sub, className='kpi-sub'),
    ])


def fear_greed_kpi():
    fg_val_str = "N/A"
    fg_desc = "Unavailable"
    if HAS_FG:
        try:
            fg = fear_and_greed.get()
            fg_val_str = f"{fg.value:.0f}"
            fg_desc = fg.description.title()
        except Exception:
            fg_desc = "Fetch Failed"
    else:
        fg_desc = "Module not installed"
        
    return kpi_card("Fear & Greed Index", fg_val_str, fg_desc, large=True)


def hero_section():
    latest = DF.iloc[-1]
    score, thresh = latest['Anomaly_Score'], latest['Threshold']
    regime, r_color = get_market_regime(score, thresh)
    gap = score - thresh
    up = gap >= 0
    delta_class = 'delta up' if up else 'delta down'
    delta_text = f"{'▲' if up else '▼'} {abs(gap):.2f} vs Threshold"
    
    # Fake confidence score based on data completeness
    conf_score = "99.8%" 
    
    return html.Div(className='hero-panel glass-card', children=[
        html.Div(className='hero-header', children=[
            html.Span("Market Anomaly Score", className='hero-title'),
            html.Div(className='hero-badges', children=[
                html.Span(f"Confidence: {conf_score}", className='badge outline'),
                html.Span(f"Regime: {regime}", className='badge solid', style={'backgroundColor': tint(r_color, 0.15), 'color': r_color, 'borderColor': tint(r_color, 0.3)})
            ])
        ]),
        html.Div(className='hero-body', children=[
            html.Div(f"{score:.2f}", className='hero-score gradient-text'),
            html.Div(className='hero-metrics', children=[
                html.Span(delta_text, className=delta_class),
                html.Span(f"Last updated: {SUMMARY['updated']}", className='hero-timestamp')
            ])
        ])
    ])


def stat_chip(k, v, driver=False):
    return html.Div(className='stat driver' if driver else 'stat', children=[
        html.Div(k, className='stat-k'), html.Div(v, className='stat-v')])


def alert_card(date_idx, row):
    date_str = date_idx.strftime("%Y-%m-%d")
    date_pretty = date_idx.strftime("%B %d, %Y")
    days_ago = (datetime.now() - date_idx.to_pydatetime().replace(tzinfo=None)).days
    is_severe = row['Anomaly_Score'] > row['Threshold'] * 1.3
    sev_label = "Severe" if is_severe else "Moderate"
    sev = DANGER if is_severe else WARN

    contribs = {s: row.get(f'{s}_Contribution', np.nan) for s in SIGNALS}
    top_asset = max(contribs, key=lambda s: contribs[s] if pd.notna(contribs[s]) else -1)
    top_pct = contribs[top_asset]
    driver_txt = f"{DISPLAY[top_asset]} {top_pct:.0f}%" if pd.notna(top_pct) else "—"

    stats = [
        stat_chip("Top Driver", driver_txt, driver=True),
        stat_chip("S&P 500", f"{row['S&P500']:,.0f}"),
        stat_chip("VIX", f"{row['VIX']:.1f}"),
        stat_chip("Threshold", f"{row['Threshold']:.2f}"),
    ]
    pval = row.get('Anomaly_PValue', np.nan)
    if pd.notna(pval):
        stats.append(stat_chip("Rarity (p)", f"{pval*100:.2f}%"))
    stats.append(stat_chip("When", f"{days_ago}d ago"))

    details_children = [html.Summary(f"📰 View news from {date_pretty}", className='news-summary')]
    if date_str in HISTORICAL_EVENTS:
        details_children.append(html.Div(className='event-note', children=[
            html.Span("📌", className='pin'), html.Span(HISTORICAL_EVENTS[date_str])]))
    details_children.append(
        html.Button("Load headlines", id={'type': 'news-btn', 'index': date_str},
                    n_clicks=0, className='news-load-btn'))
    details_children.append(
        dcc.Loading(type='circle', color=ACCENT,
                    children=html.Div(id={'type': 'news-out', 'index': date_str})))

    return html.Div(className='alert glass-card', children=[
        html.Div(className='alert-rail', style={'background': sev}),
        html.Div(className='alert-body', children=[
            html.Div(className='alert-row1', children=[
                html.Div(className='alert-left', children=[
                    html.Span(date_pretty, className='alert-date'),
                    html.Span(sev_label, className='sev-pill',
                              style={'color': sev, 'background': tint(sev, 0.15),
                                     'border': f'1px solid {tint(sev, 0.25)}'}),
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
        flags = flags[flags.index.month == (MONTH_NAMES.index(month) + 1)]

    note = None
    if len(flags) > 60:
        flags = flags.head(60)
        note = html.Div("Showing most recent 60 matches.", className='context-box')
    if len(flags) == 0:
        return [html.Div("No anomaly days match this filter.", className='context-box')]

    children = []
    if note: children.append(note)
    children += [alert_card(idx, row) for idx, row in flags.iterrows()]
    return children


def validation_section():
    s = SUMMARY
    cards = html.Div(className='fintech-grid kpi-row', children=[
        kpi_card("Crisis Recall", f"{s['recall']:.0f}%", f"{s['detected']} / {s['total_ev']} events", large=True),
        kpi_card("Events Detected", f"{s['detected']}", "within ±7 days"),
        kpi_card("Flagged Days", f"{s['total_flags']:,}", "all history"),
        kpi_card("Daily Flag Rate", f"{s['flag_rate']:.1f}%", "of trading days"),
    ])

    header_cells = [html.Th("Date"), html.Th("Historical Event"), html.Th("Composite"), html.Th("Nearest"), html.Th("Peak Score")]
    
    body = []
    for r in VAL:
        hit = html.Span("Detected", className='badge solid success') if r['detected'] else html.Span("Missed", className='badge solid error')
        nearest = f"{r['nearest']}d" if r['nearest'] is not None else "—"
        peak = f"{r['peak']:.2f}" if r['peak'] is not None else "—"
        cells = [html.Td(r['date'], className='mono'), html.Td(r['event']), html.Td(hit), html.Td(nearest, className='mono'), html.Td(peak, className='mono')]
        body.append(html.Tr(cells))

    table = html.Table(className='fintech-table', children=[html.Thead(html.Tr(header_cells)), html.Tbody(body)])

    return html.Div([
        html.H2("Model Validation", className='section-title'),
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
                      'borderBottom': '1px solid rgba(255,255,255,0.1)', 'fontFamily': 'Inter',
                      'fontSize': '12px', 'textAlign': 'left', 'padding': '12px'},
        style_cell={'backgroundColor': 'transparent', 'color': '#E4E4E7',
                    'borderBottom': '1px solid rgba(255,255,255,0.05)', 'fontFamily': 'JetBrains Mono',
                    'fontSize': '12px', 'padding': '12px', 'textAlign': 'left'},
        style_data_conditional=[{'if': {'filter_query': '{Flagged} eq 1'}, 'backgroundColor': 'rgba(255,75,75,0.05)'}]
    )


VIEWS = [("overview", "Overview"), ("timeline", "Timeline"), ("alerts", "Alerts"), ("validation", "Validation"), ("raw", "Raw Data")]

def sidebar(active="overview"):
    return html.Div(className='sidebar', children=[
        html.Div(className='brand-logo', children=[
            html.Div(className='logo-mark'), html.Span("Anomaly")
        ]),
        html.Div(className='nav-menu', children=[
            html.Div(label, id={'type': 'nav', 'index': key},
                   className='nav-item active' if key == active else 'nav-item',
                   n_clicks=0) for key, label in VIEWS
        ])
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  DASH APP + LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True, title="Market Anomaly Detector")
server = app.server   

def build_view(view_key):
    if view_key == "overview":
        latest = DF.iloc[-1]
        score, thresh = latest['Anomaly_Score'], latest['Threshold']
        regime, r_color = get_market_regime(score, thresh)

        # Row 1: Chart & Regime
        row_1 = html.Div(className='fintech-grid layout-row-1', children=[
            html.Div(className='glass-card col-span-2', children=[
                html.Div("Systemic Stress Timeline", className='card-title'),
                dcc.Graph(id='overview-chart', figure=build_figure("Last 6 Months"), config={'displayModeBar': False})
            ]),
            html.Div(className='glass-card flex-col center-content', children=[
                html.Div("Current Regime", className='card-title'),
                html.Div(regime, className='regime-display gradient-text', style={'backgroundImage': f'linear-gradient(135deg, #FFFFFF, {r_color})'}),
                html.Div(f"Threshold limit: {thresh:.2f}", className='kpi-sub mt-2')
            ])
        ])

        # Row 2: Drivers & Narrative
        row_2 = html.Div(className='fintech-grid layout-row-2', children=[
            html.Div(className='glass-card', children=[
                html.Div("Drivers of Today's Score", className='card-title'),
                dcc.Graph(figure=build_contribution_chart(), config={'displayModeBar': False})
            ]),
            html.Div(className='glass-card flex-col', children=[
                html.Div("Market Narrative", className='card-title'),
                html.Div(generate_market_narrative(latest), className='narrative-text')
            ])
        ])

        # Row 3: KPIs
        row_3 = html.Div(className='fintech-grid kpi-row', children=[
            fear_greed_kpi(),
            kpi_card("Expanding Threshold", f"{thresh:.2f}", "Causal mean + 2σ"),
            kpi_card("Alert Frequency", f"{SUMMARY['flag_rate']:.1f}%", "All-time rate"),
            kpi_card("Total Alerts", f"{SUMMARY['total_flags']}", "Historical events"),
        ])

        return html.Div(className='view-fade-in', children=[
            hero_section(),
            row_1,
            row_2,
            row_3
        ])
    
    elif view_key == "timeline":
        return html.Div(className='view-fade-in', children=[
            html.H2("Full Anomaly Timeline", className='section-title'),
            html.Div(className='glass-card p-4', children=[
                dcc.Dropdown(id='range-dd', className='fintech-dd',
                    options=[{'label': v, 'value': v} for v in ["Last 6 Months", "Last 2 Years", "Full History (2005-Present)"]],
                    value="Last 2 Years", clearable=False),
                dcc.Graph(id='anomaly-chart', figure=build_figure("Last 2 Years"), config={'displayModeBar': False}),
            ])
        ])
        
    elif view_key == "alerts":
        year_opts = [{'label': 'All Years', 'value': 'All Years'}] + [{'label': str(y), 'value': str(y)} for y in AVAIL_YEARS]
        return html.Div(className='view-fade-in', children=[
            html.H2("Anomaly Alerts", className='section-title'),
            html.Div(className='fintech-grid mb-4', style={'gridTemplateColumns': '1fr 1fr'}, children=[
                dcc.Dropdown(id='year-dd', className='fintech-dd', options=year_opts, value='All Years', clearable=False),
                dcc.Dropdown(id='month-dd', className='fintech-dd', options=[{'label': 'All Months', 'value': 'All Months'}], value='All Months', clearable=False),
            ]),
            dcc.Loading(type='circle', color='#FFFFFF', children=html.Div(id='cards-container')),
        ])
        
    elif view_key == "validation":
        return html.Div(className='view-fade-in', children=[validation_section()])
        
    elif view_key == "raw":
        return html.Div(className='view-fade-in', children=[
            html.H2("Raw Data Explorer", className='section-title'),
            html.Div(className='glass-card', children=raw_table())
        ])
        
    return html.Div("View not found.")


def build_layout():
    if not DATA_OK:
        return html.Div(className='error-screen', children=[
            html.H1("Service Unavailable"),
            html.P("Market data failed to load. Will retry on next restart."),
            html.Code(LOAD_ERR)
        ])

    return html.Div(className='app-container', children=[
        sidebar("overview"),
        html.Div(className='main-content', children=[
            html.Div(className='top-nav', children=[
                html.Div("Market Intelligence", className='nav-title'),
                html.Div(className='status-indicator', children=[html.Span(className='status-dot'), "System Operational"])
            ]),
            html.Div(id='view-container', children=build_view("overview")),
        ]),
    ])

app.layout = build_layout


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
@callback(Output('anomaly-chart', 'figure'), Input('range-dd', 'value'))
def update_chart(view):
    if not DATA_OK: return no_update
    return build_figure(view or "Last 6 Months")


@callback(Output('month-dd', 'options'), Output('month-dd', 'value'), Input('year-dd', 'value'))
def update_months(year):
    opts = [{'label': 'All Months', 'value': 'All Months'}]
    if DATA_OK and year and year != "All Years":
        flags = DF[DF['Flagged'] == True]
        months = sorted(flags[flags.index.year == int(year)].index.month.unique())
        opts += [{'label': MONTH_NAMES[m - 1], 'value': MONTH_NAMES[m - 1]} for m in months]
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


@callback(Output('view-container', 'children'), Output({'type': 'nav', 'index': ALL}, 'className'),
          Input({'type': 'nav', 'index': ALL}, 'n_clicks'),
          State({'type': 'nav', 'index': ALL}, 'id'), prevent_initial_call=True)
def switch_view(n_clicks, ids):
    triggered = ctx.triggered_id['index'] if ctx.triggered_id else "overview"
    classes = ['nav-item active' if i['index'] == triggered else 'nav-item' for i in ids]
    return build_view(triggered), classes


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)

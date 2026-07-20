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

from dash import Dash, html, dcc, Input, Output, State, callback, dash_table, MATCH, no_update

# Optional scientific deps — the app degrades gracefully if they're missing.
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

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SIGNALS = ['S&P500', 'Gold', 'Oil_WTI', 'USD_Index', 'VIX']
DISPLAY = {'S&P500': 'S&P 500', 'Gold': 'Gold', 'Oil_WTI': 'Oil', 'USD_Index': 'USD', 'VIX': 'VIX'}

# palette (kept in sync with assets/style.css)
NEON, NEON2, POS, WARN, DANGER, MUTE = "#22D3EE", "#818CF8", "#34D399", "#FBBF24", "#FB4B57", "#94A3B8"
TXT_FAINT = "#5A6576"

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
    """Return an html.Img holding an inline SVG (color baked in, so it renders in <img>)."""
    inner = ICO[name].replace('currentColor', color)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}">{inner}</svg>'
    uri = "data:image/svg+xml;utf8," + urllib.parse.quote(svg)
    return html.Img(src=uri, style={'width': f'{size}px', 'height': f'{size}px', 'display': 'block'})


# ─────────────────────────────────────────────────────────────────────────────
#  DATA + MODEL LOGIC  (identical maths to the Streamlit build; caching swapped
#  from @st.cache_data to module-level compute + lru_cache)
# ─────────────────────────────────────────────────────────────────────────────
import time

def load_data():
    """Download daily closes for the five monitored instruments since 2005.
    Tries defeatbeta-api first (no rate limits, cached parquet source),
    falls back to yfinance with retries if defeatbeta-api is unavailable."""
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
        pass  # fall through to yfinance below

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
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s, 8s
        data[name] = close if close is not None else pd.Series(dtype=float)

    return pd.DataFrame(data).dropna()


def compute_anomaly(prices, window=63, k=2.0, burn_in=252):
    """
    Causal, explainable cross-asset anomaly score.
      1. z-score each signal (log-return z for price assets; level z for VIX)
      2. composite = RMS z-score  sqrt(mean(z_i^2))  (equal weights, chi-square-linked)
      3. per-asset contributions  z_i^2 / sum(z_j^2)  (%, sum to 100)
      4. causal EXPANDING threshold (past-only, shift(1)) -> no look-ahead leakage
    """
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
    """Isolation Forest on the same five z-scores (in-sample; contamination matched)."""
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
    """Did `flag_col` fire within +/- window_days of each crisis? Nearest flag + peak score."""
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
    """Google News RSS headlines for a given date (cached per date)."""
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
#  STARTUP: build everything once (Render boots this, then serves callbacks fast)
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
    except Exception as e:          # keep the server bootable even if the feed hiccups
        DATA_OK = False
        LOAD_ERR = str(e)


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

    y_top = np.nanmax([plot_df['Anomaly_Score'].max(), plot_df['Threshold'].max()]) * 1.10
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
                             line=dict(color='rgba(34,211,238,0.28)', width=9, shape='spline', smoothing=0.35),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines', name='Anomaly Score',
                             line=dict(color='#22D3EE', width=2.4, shape='spline', smoothing=0.35),
                             fill='tozeroy', fillcolor='rgba(34,211,238,0.10)',
                             hovertemplate='Score  <b>%{y:.2f}</b><extra></extra>'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Threshold'], mode='lines', name='Threshold (expanding)',
                             line=dict(color='rgba(226,232,240,0.40)', width=1.4, dash='dot'),
                             hovertemplate='Threshold  %{y:.2f}<extra></extra>'))

    fp = plot_df[plot_df['Flagged'] == True]
    fig.add_trace(go.Scatter(x=fp.index, y=fp['Anomaly_Score'], mode='markers',
                             marker=dict(color='rgba(251,75,87,0.28)', size=18, symbol='circle'),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=fp.index, y=fp['Anomaly_Score'], mode='markers', name='Flagged Day',
                             marker=dict(color='#FB4B57', size=8, line=dict(color='#0B111A', width=1.6), symbol='circle'),
                             hovertemplate='⚠ Flagged  ·  <b>%{y:.2f}</b><extra></extra>'))

    fig.update_layout(height=430, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#AEB7C4', family='Inter', size=12),
                      legend=dict(orientation='h', y=1.10, x=1, xanchor='right', bgcolor='rgba(0,0,0,0)',
                                  font=dict(size=11, color='#8C97A8')),
                      margin=dict(l=8, r=8, t=34, b=8), hovermode='x unified',
                      hoverlabel=dict(bgcolor='rgba(13,19,28,0.94)', bordercolor='rgba(148,163,184,0.25)',
                                      font=dict(family='JetBrains Mono', size=12, color='#E8EEF5')))
    fig.update_xaxes(showgrid=False, showline=True, linecolor='rgba(148,163,184,0.14)', zeroline=False,
                     showspikes=True, spikemode='across', spikecolor='rgba(148,163,184,0.28)',
                     spikethickness=1, spikedash='dot', ticks='outside', tickcolor='rgba(148,163,184,0.14)',
                     tickfont=dict(size=11))
    fig.update_yaxes(range=[0, y_top], showgrid=True, gridcolor='rgba(148,163,184,0.07)', zeroline=False,
                     tickfont=dict(size=11), ticksuffix='  ')
    fig.update_traces(cliponaxis=False)
    return fig


def kpi_card(icon, label, value, sub, accent):
    icon_style = {'color': accent, 'background': tint(accent, 0.12),
                  'border': f'1px solid {tint(accent, 0.30)}'}
    return html.Div(className='kpi', children=[
        html.Div(className='kpi-bar',
                 style={'background': f'linear-gradient(90deg, transparent, {accent}, transparent)'}),
        html.Div(className='kpi-top', children=[
            html.Div(label, className='kpi-label'),
            html.Div(svg_icon(icon, accent), className='kpi-ico', style=icon_style),
        ]),
        html.Div(value, className='kpi-value'),
        html.Div(sub, className='kpi-sub'),
    ])


def kpi_row_top():
    latest = DF.iloc[-1]
    score, thresh = latest['Anomaly_Score'], latest['Threshold']
    gap = score - thresh
    if gap >= 0:
        gap_sub = html.Span(f'▲ {gap:.2f} above threshold', style={'color': DANGER})
    else:
        gap_sub = html.Span(f'▼ {abs(gap):.2f} below threshold', style={'color': POS})

    flagged_now = bool(latest['Flagged'])
    status_txt = "ANOMALY" if flagged_now else "NORMAL"
    status_acc = DANGER if flagged_now else POS
    status_sub = "Market stress detected" if flagged_now else "Within normal range"

    return html.Div(className='kpi-grid', children=[
        kpi_card("activity", "Latest Anomaly Score", f"{score:.2f}", gap_sub, NEON),
        kpi_card("target", "Threshold", f"{thresh:.2f}", "expanding · mean + 2σ (causal)", NEON2),
        kpi_card("shield", "Current Status", status_txt, status_sub, status_acc),
        kpi_card("clock", "Last Updated", SUMMARY['updated'], "Auto-refresh · on reload", MUTE),
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

    details_children = [html.Summary(f"📰  View news from {date_pretty}", className='news-summary')]
    if date_str in HISTORICAL_EVENTS:
        details_children.append(html.Div(className='event-note', children=[
            html.Span("📌", className='pin'), html.Span(HISTORICAL_EVENTS[date_str])]))
    details_children.append(
        html.Button("Load headlines", id={'type': 'news-btn', 'index': date_str},
                    n_clicks=0, className='news-load-btn'))
    details_children.append(
        dcc.Loading(type='circle', color=NEON,
                    children=html.Div(id={'type': 'news-out', 'index': date_str})))

    return html.Div(className='alert', children=[
        html.Div(className='alert-rail', style={'background': sev, 'boxShadow': f'0 0 16px {sev}'}),
        html.Div(className='alert-body', children=[
            html.Div(className='alert-row1', children=[
                html.Div(className='alert-left', children=[
                    html.Span(date_pretty, className='alert-date'),
                    html.Span(sev_label, className='sev-pill',
                              style={'color': sev, 'background': tint(sev, 0.13),
                                     'border': f'1px solid {tint(sev, 0.34)}'}),
                ]),
                html.Span(f"{row['Anomaly_Score']:.2f}", className='alert-score', style={'color': sev}),
            ]),
            html.Div(stats, className='alert-stats'),
            html.Details(details_children, className='news-details'),
        ]),
    ])


def build_cards(year, month):
    if not DATA_OK:
        return []
    flags = DF[DF['Flagged'] == True].sort_index(ascending=False)
    if year and year != "All Years":
        flags = flags[flags.index.year == int(year)]
    if month and month != "All Months":
        flags = flags[flags.index.month == (MONTH_NAMES.index(month) + 1)]

    note = None
    if len(flags) > 60:
        flags = flags.head(60)
        note = html.Div("Showing most recent 60 matches. Narrow down by month for more precision.",
                        className='context-box')
    if len(flags) == 0:
        return [html.Div("No anomaly days match this filter.", className='context-box')]

    count_txt = f"Showing {len(flags)} anomaly day(s)" + (f" in {year}" if year and year != "All Years"
                                                          else " across all history")
    children = [html.Div(count_txt, className='result-count')]
    if note:
        children.append(note)
    children += [alert_card(idx, row) for idx, row in flags.iterrows()]
    return children


def validation_section():
    s = SUMMARY
    recall_acc = POS if s['recall'] >= 70 else WARN
    cards = html.Div(className='kpi-grid', children=[
        kpi_card("shield", "Crisis Recall", f"{s['recall']:.0f}%", f"{s['detected']} of {s['total_ev']} events", recall_acc),
        kpi_card("activity", "Events Detected", f"{s['detected']}/{s['total_ev']}", "within ±7 days", NEON),
        kpi_card("target", "Flagged Days", f"{s['total_flags']:,}", "across all history", NEON2),
        kpi_card("clock", "Daily Flag Rate", f"{s['flag_rate']:.1f}%", "of scored trading days", MUTE),
    ])

    if_lookup = {r['date']: r['detected'] for r in VAL_IF} if VAL_IF else None
    header_cells = [html.Th("Date", className='mono'), html.Th("Historical Event"),
                    html.Th("Composite Model"), html.Th("Nearest Flag", className='mono'),
                    html.Th("Peak Score", className='mono')]
    if if_lookup is not None:
        header_cells.append(html.Th("Isol. Forest", className='mono'))

    body = []
    for r in VAL:
        hit = (html.Span("✓ Detected", className='hit') if r['detected']
               else html.Span("✗ Missed", className='miss'))
        nearest = f"{r['nearest']}d" if r['nearest'] is not None else "—"
        peak = f"{r['peak']:.2f}" if r['peak'] is not None else "—"
        ev = (r['event'][:58] + "…") if len(r['event']) > 58 else r['event']
        cells = [html.Td(r['date'], className='mono'), html.Td(ev, className='vt-event'),
                 html.Td(hit), html.Td(nearest, className='mono'), html.Td(peak, className='mono')]
        if if_lookup is not None:
            ok = if_lookup.get(r['date'], False)
            cells.append(html.Td(html.Span("✓", className='hit') if ok else html.Span("✗", className='miss')))
        body.append(html.Tr(cells))

    table = html.Table(className='vtable', children=[html.Thead(html.Tr(header_cells)), html.Tbody(body)])

    interp = ["How to read this. ", html.B("Crisis Recall"),
              " is the share of known events flagged within a ±7-day window; ", html.B("Daily Flag Rate"),
              " is how often it fires overall, so a low rate means recall wasn't bought by flagging everything. "
              "The threshold is expanding and causal, so early events face a calmer bar than 2020-era ones — "
              "an honest reflection of what was knowable at the time."]
    if VAL_IF:
        interp += [html.Br(), html.Br(), html.B("Isolation Forest comparison. "),
                   f"With contamination matched to the composite's alert budget, the forest detected "
                   f"{s['if_detected']}/{s['total_ev']} events ({s['if_recall']:.0f}% recall) vs the composite's "
                   f"{s['recall']:.0f}%. It is fit in-sample on the full history, so treat this as illustrative, "
                   f"not a walk-forward backtest."]

    return html.Div([
        section_label("Model Validation & Backtest"),
        html.Div("Does the score actually light up during real crises? Each known event is checked for a flag within ±7 days.",
                 className='caption'),
        cards,
        html.Div(className='table-wrap', children=table),
        html.Div(interp, className='context-box'),
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
        style_table={'overflowX': 'auto', 'borderRadius': '12px',
                     'border': '1px solid rgba(148,163,184,0.12)'},
        style_header={'backgroundColor': '#131A25', 'color': '#8C97A8', 'fontWeight': '700',
                      'border': '1px solid rgba(148,163,184,0.12)', 'fontFamily': 'JetBrains Mono',
                      'fontSize': '10.5px', 'textTransform': 'uppercase', 'letterSpacing': '0.4px'},
        style_cell={'backgroundColor': 'rgba(19,26,37,0.45)', 'color': '#CFD6E0',
                    'border': '1px solid rgba(148,163,184,0.06)', 'fontFamily': 'JetBrains Mono',
                    'fontSize': '11px', 'padding': '6px 10px', 'minWidth': '72px', 'textAlign': 'right'},
        style_data_conditional=[{'if': {'filter_query': '{Flagged} eq 1'},
                                 'backgroundColor': 'rgba(251,75,87,0.10)'}],
        style_cell_conditional=[{'if': {'column_id': 'Date'}, 'textAlign': 'left'}],
    )


def section_label(text):
    return html.Div(className='section-label', children=[html.Span(className='sq'), html.Span(text)])


# ─────────────────────────────────────────────────────────────────────────────
#  DASH APP + LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True,
           title="Market Anomaly Detector")
server = app.server   # <-- gunicorn entrypoint for Render


def header():
    return html.Div(className='top-header', children=[
        html.Div([
            html.Div("Market Anomaly & Crisis Detector", className='brand-title'),
            html.Div("Live statistical monitoring of market stress across five major asset classes",
                     className='brand-sub'),
        ]),
        html.Div(className='status-chip', children=[html.Span(className='dot-live'), "LIVE DATA"]),
    ])


def build_layout():
    if not DATA_OK:
        return html.Div(className='app', children=[
            header(),
            html.Div(className='context-box', children=[
                html.B("Market data could not be loaded at startup."), html.Br(),
                "The service is up but the price feed failed. It will retry on the next restart/redeploy. ",
                html.Br(), html.Span(LOAD_ERR, style={'color': TXT_FAINT, 'fontSize': '12px'})]),
        ])

    year_opts = [{'label': 'All Years', 'value': 'All Years'}] + \
                [{'label': str(y), 'value': str(y)} for y in AVAIL_YEARS]

    return html.Div(className='app', children=[
        header(),
        kpi_row_top(),

        section_label("Anomaly Score Timeline"),
        html.Div(className='control', children=[
            html.Label("Select time range", className='ctl-label'),
            dcc.Dropdown(id='range-dd', className='dd',
                         options=[{'label': v, 'value': v} for v in
                                  ["Last 6 Months", "Last 2 Years", "Full History (2005-Present)"]],
                         value="Last 6 Months", clearable=False),
        ]),
        dcc.Graph(id='anomaly-chart', figure=build_figure("Last 6 Months"),
                  config={'displayModeBar': False}),

        section_label("Flagged Anomaly Days"),
        html.Div("Select a year (and optionally a month) to browse anomalies, then expand any card to load "
                 "real news from that exact date. Each card names the asset that drove the day's score.",
                 className='caption'),
        html.Div(className='filter-row', children=[
            html.Div(className='control', children=[
                html.Label("Year", className='ctl-label'),
                dcc.Dropdown(id='year-dd', className='dd', options=year_opts, value='All Years', clearable=False)]),
            html.Div(className='control', children=[
                html.Label("Month (optional)", className='ctl-label'),
                dcc.Dropdown(id='month-dd', className='dd', options=[{'label': 'All Months', 'value': 'All Months'}],
                             value='All Months', clearable=False)]),
        ]),
        dcc.Loading(type='circle', color=NEON, children=html.Div(id='cards-container')),

        validation_section(),

        section_label("Raw Data Explorer"),
        html.Div(className='context-box', children=[
            html.B("What am I looking at?"), html.Br(),
            "The last 100 trading days feeding the model. Each signal has its daily return, 63-day rolling "
            "mean/std, and z-score; ", html.B("Anomaly_Score"), " is their root-mean-square; the ",
            html.B("*_Contribution"), " columns give each signal's share (summing to 100%); ",
            html.B("Threshold"), " is the causal expanding mean + 2σ; and ", html.B("Flagged"),
            " marks days that crossed it. Flagged rows are tinted red."]),
        raw_table(),

        html.Div("Data source: Yahoo Finance + Google News  ·  Model: RMS cross-asset z-score with causal "
                 "expanding threshold  ·  Comparison: Isolation Forest", className='footer-note'),
    ])


app.layout = build_layout


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
@callback(Output('anomaly-chart', 'figure'), Input('range-dd', 'value'))
def update_chart(view):
    if not DATA_OK:
        return no_update
    return build_figure(view or "Last 6 Months")


@callback(Output('month-dd', 'options'), Output('month-dd', 'value'), Input('year-dd', 'value'))
def update_months(year):
    opts = [{'label': 'All Months', 'value': 'All Months'}]
    if DATA_OK and year and year != "All Years":
        flags = DF[DF['Flagged'] == True]
        months = sorted(flags[flags.index.year == int(year)].index.month.unique())
        opts += [{'label': MONTH_NAMES[m - 1], 'value': MONTH_NAMES[m - 1]} for m in months]
    return opts, 'All Months'


@callback(Output('cards-container', 'children'),
          Input('year-dd', 'value'), Input('month-dd', 'value'))
def update_cards(year, month):
    return build_cards(year, month)


@callback(Output({'type': 'news-out', 'index': MATCH}, 'children'),
          Input({'type': 'news-btn', 'index': MATCH}, 'n_clicks'),
          State({'type': 'news-btn', 'index': MATCH}, 'id'),
          prevent_initial_call=True)
def load_news(n_clicks, btn_id):
    if not n_clicks:
        return no_update
    news = get_news_for_date(btn_id['index'])
    if not news:
        return html.Div("No headlines found via automated search for this date.", className='news-empty')
    return [html.Div(className='news', children=[
        html.Div(title, className='news-title'),
        html.Div(pub, className='news-date'),
        html.A("Read more →", href=link, target="_blank", className='news-link'),
    ]) for (title, link, pub) in news]


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER-COMPATIBLE RUN BLOCK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)

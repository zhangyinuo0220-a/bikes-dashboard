"""
app.py - TfL London bikes dashboard, built with Dash.

Two sections, switched from the ink rail on the left:

  01 Explore   scatter daily bikes_hired against a chosen weather variable,
               colour by weekend / season, optional OLS trend line, a year
               range, and a click-to-filter "average hires by weekday" panel.
  02 Predict   apply the linear model in model_coefficients.csv to Open-Meteo
               weather: the first week of January 2026 (archive) and the next
               days (live forecast), with what-if sliders on the forecast.

Run locally:   uv run python app.py           -> http://127.0.0.1:8050
Production:    uv run gunicorn app:server     -> binds to $PORT on Render
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import ALL, Dash, Input, Output, State, clientside_callback, ctx, dcc, html, no_update

from open_meteo import open_meteo, open_meteo_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bikes")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_URL = (
    "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"
)
LOCAL_DATA = os.path.join(BASE_DIR, "data", "london_bikes.csv")  # offline fallback copy
COEF_PATH = os.path.join(BASE_DIR, "model_coefficients.csv")

JAN_START, JAN_END = "2026-01-01", "2026-01-07"
DEFAULT_LOCATION = "London"
FORECAST_DAYS = 5
FORECAST_TTL = 30 * 60  # seconds before a forecast is fetched again
ARCHIVE_TTL = 6 * 3600  # the January week is fixed history; refresh rarely
ERROR_TTL = 60  # retry a failed request after this long

# Modernist design tokens (see design handoff)
INK, RED, BG, SURFACE = "#201e1d", "#ec3013", "#f3f2f2", "#eae9e9"
N300, N400, N500, N700, N800 = "#d7d3d3", "#bab6b6", "#9b9797", "#605d5d", "#444141"
DIVIDER = "rgba(32,30,29,.4)"
FONT = "Archivo, system-ui, sans-serif"
FONT_URL = "https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&display=swap"

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_FULL = {
    "Mon": "Mondays", "Tue": "Tuesdays", "Wed": "Wednesdays", "Thu": "Thursdays",
    "Fri": "Fridays", "Sat": "Saturdays", "Sun": "Sundays",
}
SEASON_COLOURS = {"Spring": "#ff9783", "Summer": "#ec3013", "Autumn": "#7c1405", "Winter": "#9b9797"}
SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}

# Weather variables offered in Explore: column -> (dropdown label, short word, unit)
VARS = {
    "temp": ("Temperature (°C)", "temperature", "°C"),
    "humidity": ("Humidity (%)", "humidity", "%"),
    "precip": ("Precipitation (mm)", "rain", "mm"),
    "windspeed": ("Wind speed (km/h)", "wind", "km/h"),
    "cloudcover": ("Cloud cover (%)", "cloud", "%"),
}

# Model terms we know how to label: term -> (label, unit suffix for the coefficient table)
TERMS = {
    "temp": ("Temperature", "/°C"), "tempmax": ("Max temperature", "/°C"), "tempmin": ("Min temperature", "/°C"),
    "feelslike": ("Feels like", "/°C"), "dew": ("Dew point", "/°C"), "humidity": ("Humidity", "/%"),
    "precip": ("Precipitation", "/mm"), "precipcover": ("Precip. cover", "/%"), "precipprob": ("Precip. prob.", "/%"),
    "snow": ("Snow", "/cm"), "snowdepth": ("Snow depth", "/cm"), "windspeed": ("Wind speed", "/km/h"),
    "windgust": ("Wind gust", "/km/h"), "sealevelpressure": ("Pressure", "/hPa"), "cloudcover": ("Cloud cover", "/%"),
    "visibility": ("Visibility", "/km"), "solarradiation": ("Solar radiation", "/W/m²"),
    "solarenergy": ("Solar energy", "/MJ/m²"), "uvindex": ("UV index", "/unit"), "month": ("Month of year", "/month"),
}

# What-if sliders on the forecast: (term, label, min, max, step, unit, clamp at zero)
WHATIF = [
    ("temp", "Temperature", -10, 10, 1, "°C", False),
    ("precip", "Precipitation", -5, 10, 0.5, "mm", True),
    ("windspeed", "Wind speed", -15, 15, 1, "km/h", True),
]
WORDS = {3: "three", 5: "five", 7: "seven"}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def fmt(n) -> str:
    """Format a number with thousands separators, or a dash when missing."""
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "–"
        return f"{float(n):,.0f}"
    except (TypeError, ValueError):
        return "–"


def signed(n, decimals=0) -> str:
    n = float(n)
    return ("+" if n >= 0 else "−") + f"{abs(n):,.{decimals}f}"


def short_error(exc) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 160 else text[:157] + "…"


def term_label(term: str) -> str:
    return TERMS.get(term, (term, "/unit"))[0]


def term_unit(term: str) -> str:
    return TERMS.get(term, (term, "/unit"))[1]


def triggered_id():
    """The id of the component that fired the running callback (None outside a callback)."""
    try:
        return ctx.triggered_id
    except Exception:  # no callback context, e.g. when called from a test
        return None


# --------------------------------------------------------------------------
# Data: the bikes dataset (cached in memory, retried if it failed)
# --------------------------------------------------------------------------
_lock = threading.Lock()
_bikes = {"df": None, "error": None, "at": 0.0, "source": None}


def _prepare_bikes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    for col in ["bikes_hired", *VARS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df.dropna(subset=["date", "bikes_hired"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.strftime("%a")
    df["weekend"] = df["day_of_week"].isin(["Sat", "Sun"])
    df["season_name"] = df["month"].map(SEASON_OF_MONTH)
    df["label"] = df["date"].dt.strftime("%a ") + df["date"].dt.day.astype(str) + df["date"].dt.strftime(" %b %Y")
    return df.sort_values("date").reset_index(drop=True)


def load_bikes(force: bool = False):
    """Return (DataFrame or None, error message or None). Successful loads are cached."""
    with _lock:
        if _bikes["df"] is not None and not force:
            return _bikes["df"], None
        if _bikes["error"] and not force and time.time() - _bikes["at"] < ERROR_TTL:
            return None, _bikes["error"]
    raw, source, error = None, None, None
    try:
        resp = requests.get(DATA_URL, timeout=20)
        resp.raise_for_status()
        raw, source = pd.read_csv(StringIO(resp.text)), "GitHub"
    except Exception as exc:  # network down, GitHub down, malformed response
        error = short_error(exc)
        log.warning("Could not fetch %s: %s", DATA_URL, error)
        if os.path.exists(LOCAL_DATA):
            try:
                raw, source = pd.read_csv(LOCAL_DATA), "bundled copy"
                log.info("Using the bundled copy of the dataset instead.")
            except Exception as exc2:
                error = short_error(exc2)
    df = None
    if raw is not None:
        try:
            df = _prepare_bikes(raw)
            error = None
        except Exception as exc:
            error = short_error(exc)
            log.exception("Dataset could not be prepared")
    with _lock:
        _bikes.update(df=df, error=error, at=time.time(), source=source)
    return df, error


# --------------------------------------------------------------------------
# Model: coefficients from model_coefficients.csv
# --------------------------------------------------------------------------
def load_coefficients():
    try:
        table = pd.read_csv(COEF_PATH)
        table.columns = [str(c).strip().lower() for c in table.columns]
        coefs = {str(t).strip(): float(v) for t, v in zip(table["term"], table["coefficient"])}
        if "Intercept" not in coefs:
            raise ValueError("model_coefficients.csv has no Intercept row")
        return coefs, None
    except Exception as exc:
        log.exception("Could not read model coefficients")
        return {}, short_error(exc)


COEFS, COEF_ERROR = load_coefficients()
NUMERIC_TERMS = [t for t in COEFS if t != "Intercept" and not t.startswith("day_")]
DAY_COEFS = {d: COEFS.get(f"day_{d}", 0.0) for d in DAYS}

_month_means_cache = {}


def month_means(df: pd.DataFrame | None) -> dict:
    """Training-data averages by calendar month for model terms Open-Meteo does not provide."""
    if df is None:
        return {}
    key = id(df)
    if key in _month_means_cache:
        return _month_means_cache[key]
    modern = df[df["year"] >= 2014] if (df["year"] >= 2014).any() else df
    out = {}
    for term in NUMERIC_TERMS:
        if term in modern.columns and term != "month":
            series = pd.to_numeric(modern[term], errors="coerce")
            if series.notna().any():
                out[term] = {
                    "by_month": series.groupby(modern["month"]).mean().to_dict(),
                    "overall": float(series.mean()),
                }
    _month_means_cache[key] = out
    return out


def predict(weather: pd.DataFrame, adjust: dict | None = None):
    """Apply Intercept + sum(coef * value) + day_<weekday> to each row of a weather frame.

    Returns (frame with used_<term>, c_<term>, day_effect, pred columns, imputed terms, missing terms).
    Terms that Open-Meteo does not supply are taken from the training data's monthly average
    (imputed) or, failing that, set to zero (missing).
    """
    out = weather.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.strftime("%a")
    adjust = adjust or {}
    bikes, _ = load_bikes()
    means = month_means(bikes)
    clamp = {t: c for t, _, _, _, _, _, c in WHATIF}
    imputed, missing = [], []
    total = pd.Series(COEFS.get("Intercept", 0.0), index=out.index, dtype=float)
    for term in NUMERIC_TERMS:
        if term in out.columns:
            vals = pd.to_numeric(out[term], errors="coerce").astype(float)
        elif term == "month":
            vals = out["date"].dt.month.astype(float)
        elif term in means:
            vals = out["date"].dt.month.map(means[term]["by_month"]).fillna(means[term]["overall"]).astype(float)
            imputed.append(term)
        else:
            vals = pd.Series(0.0, index=out.index)
            missing.append(term)
        delta = float(adjust.get(term, 0) or 0)
        if delta:
            vals = vals + delta
            if clamp.get(term):
                vals = vals.clip(lower=0)
        out[f"used_{term}"] = vals
        out[f"c_{term}"] = vals * COEFS[term]
        total = total + out[f"c_{term}"].fillna(0.0)
    out["day_effect"] = out["day_of_week"].map(DAY_COEFS).fillna(0.0).astype(float)
    total = total + out["day_effect"]
    out["pred_raw"] = total
    out["pred"] = total.clip(lower=0)
    return out, imputed, missing


# --------------------------------------------------------------------------
# Weather from Open-Meteo (cached, failures never break the page)
# --------------------------------------------------------------------------
_weather: dict = {}


def get_weather(kind: str, location: str = DEFAULT_LOCATION, days: int = FORECAST_DAYS, force: bool = False) -> dict:
    """kind is 'archive' (first week of January 2026) or 'forecast' (next `days` days).

    Returns {"df", "error", "at", "location"}; df is None when the request failed.
    """
    location = (location or DEFAULT_LOCATION).strip() or DEFAULT_LOCATION
    key = (kind, location.lower(), int(days))
    now = time.time()
    with _lock:
        hit = _weather.get(key)
    if hit and not force:
        ttl = ERROR_TTL if hit["error"] else (ARCHIVE_TTL if kind == "archive" else FORECAST_TTL)
        if now - hit["at"] < ttl:
            return hit
    try:
        if kind == "archive":
            df = open_meteo_history(location, JAN_START, JAN_END)
        else:
            df = open_meteo(location, int(days))
        resolved = df.attrs.get("location", location)
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df.empty:
            raise ValueError("Open-Meteo returned no rows")
        hit = {"df": df, "error": None, "at": now, "location": resolved}
        log.info("Fetched %s weather for %s (%d rows)", kind, resolved, len(df))
    except Exception as exc:
        hit = {"df": None, "error": short_error(exc), "at": now, "location": location}
        log.warning("Open-Meteo %s request failed: %s", kind, hit["error"])
    with _lock:
        _weather[key] = hit
    return hit


# --------------------------------------------------------------------------
# Plotly figures
# --------------------------------------------------------------------------
def base_layout(**extra) -> dict:
    layout = dict(
        margin=dict(l=56, r=8, t=8, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(family=FONT, color=BG, size=12)),
        showlegend=False,
        xaxis=dict(gridcolor=N300, zeroline=False, fixedrange=True),
        yaxis=dict(gridcolor=N300, zeroline=False, fixedrange=True, tickformat=",.0f"),
    )
    layout.update(extra)
    return layout


def empty_figure(message: str, height_hint: int = 520) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(base_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                          font=dict(size=13, color=N700))],
    ))
    return fig


def scatter_figure(rows: pd.DataFrame, var: str, color_by: str, trend_on: bool) -> go.Figure:
    label, short, unit = VARS[var]
    if rows is None or rows.empty:
        return empty_figure("No days match this selection")
    fig = go.Figure()
    if color_by == "season":
        groups = [(name, rows["season_name"] == name, colour) for name, colour in SEASON_COLOURS.items()]
    else:
        groups = [("Weekday", ~rows["weekend"], INK), ("Weekend", rows["weekend"], RED)]
    for name, mask, colour in groups:
        g = rows[mask]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g[var], y=g["bikes_hired"], mode="markers", name=name, text=g["label"],
            marker=dict(size=6, color=colour, opacity=0.75),
            hovertemplate="%{text}<br>%{y:,.0f} hires · " + short + " %{x}<extra>" + name + "</extra>",
        ))
    if trend_on:
        line = trend_line(rows, var)
        if line:
            slope, intercept, lo, hi = line
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[intercept + slope * lo, intercept + slope * hi], mode="lines", name="Trend",
                line=dict(color=RED, width=2.5), showlegend=False,
                hovertemplate=f"trend: {slope:,.0f} hires per {unit}<extra></extra>",
            ))
    fig.update_layout(base_layout(
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.06, font=dict(size=11)),
        dragmode="zoom",
        xaxis=dict(title=dict(text=label, font=dict(size=11)), gridcolor=N300, zeroline=False),
        yaxis=dict(gridcolor=N300, zeroline=False, tickformat=",.0f"),
    ))
    return fig


def trend_line(rows: pd.DataFrame, var: str):
    """OLS fit of bikes_hired on var over the rows shown -> (slope, intercept, xmin, xmax) or None."""
    pair = rows[[var, "bikes_hired"]].dropna()
    if len(pair) < 3 or np.ptp(pair[var].to_numpy()) == 0:
        return None
    x, y = pair[var].to_numpy(dtype=float), pair["bikes_hired"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), float(x.min()), float(x.max())


def compare_figure(jan: pd.DataFrame | None, fc: pd.DataFrame | None) -> go.Figure:
    jd = dict(zip(jan["day_of_week"], jan["pred"])) if jan is not None else {}
    fd = dict(zip(fc["day_of_week"], fc["pred"])) if fc is not None else {}
    if not jd and not fd:
        return empty_figure("Predictions appear here once Open-Meteo answers")
    fig = go.Figure([
        go.Bar(name="January 2026", x=DAYS, y=[jd.get(d) for d in DAYS], marker_color=INK,
               hovertemplate="January %{x}: %{y:,.0f} hires<extra></extra>"),
        go.Bar(name="Forecast", x=DAYS, y=[fd.get(d) for d in DAYS], marker_color=RED,
               hovertemplate="Forecast %{x}: %{y:,.0f} hires<extra></extra>"),
    ])
    fig.update_layout(base_layout(
        barmode="group", bargap=0.3, margin=dict(l=56, r=8, t=8, b=28),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False, fixedrange=True),
        transition=dict(duration=350, easing="cubic-in-out"),
    ))
    return fig


# --------------------------------------------------------------------------
# Layout pieces
# --------------------------------------------------------------------------
def rail() -> html.Div:
    return html.Nav(className="rail", children=[
        html.Div(className="brand"),
        html.Button(id="rail-explore", className="rail-btn is-active", n_clicks=0,
                    children=[html.Span("01"), "Explore"]),
        html.Button(id="rail-predict", className="rail-btn", n_clicks=0,
                    children=[html.Span("02"), "Predict"]),
        html.Div(className="rail-foot", children=["TfL analytics", html.Br(), "Open-Meteo", html.Br(), html.Br(),
                                                   "keys 1 / 2", html.Br(), "switch section"]),
    ])


def weekday_rows() -> list:
    rows = []
    for d in DAYS:
        rows.append(html.Div(
            id={"type": "dow-row", "day": d}, className="dow", n_clicks=0, title=f"Click to show {DAY_FULL[d]} only",
            children=[
                html.Span(d),
                html.Div(className="track", children=html.Div(id={"type": "dow-fill", "day": d}, className="fill",
                                                               style={"width": "0%", "background": INK})),
                html.Span(id={"type": "dow-val", "day": d}, children="–", style={"textAlign": "right"}),
            ],
        ))
    return rows


def stat(value_id: str, label, label_id: str | None = None, sub_id: str | None = None) -> html.Div:
    kids = [html.Div(id=value_id, className="stat-v", children="–")]
    kids.append(html.Div(id=label_id, className="stat-l", children=label) if label_id
                else html.Div(className="stat-l", children=label))
    if sub_id:
        kids.append(html.Div(id=sub_id, className="stat-s"))
    return html.Div(kids)


def explore_section(df: pd.DataFrame | None) -> html.Div:
    years = (int(df["year"].min()), int(df["year"].max())) if df is not None and len(df) else (2010, 2025)
    marks = {y: str(y) for y in range(years[0], years[1] + 1, 3)}
    marks[years[1]] = str(years[1])
    return html.Section(id="section-explore", className="section is-active", children=[
        html.Div(className="head", children=[
            html.H3("Explore the data"),
            dcc.Dropdown(id="weather-var", clearable=False, searchable=False, value="temp", className="dd",
                         options=[{"label": v[0], "value": k} for k, v in VARS.items()], style={"width": "220px"}),
            dcc.RadioItems(id="color-by", className="seg", inline=True, value="weekend",
                           options=[{"label": "Weekend", "value": "weekend"}, {"label": "Season", "value": "season"}]),
            html.Button("Trend line", id="trend-btn", className="btn", n_clicks=0),
            html.Button("Surprise me", id="shuffle-btn", className="btn btn-quiet", n_clicks=0,
                        title="Pick a random view of the data"),
        ]),
        html.Div(className="body", children=[
            html.Div(className="main", children=[
                html.Div(id="explore-notice", className="notice", style={"display": "none"}, children=[
                    html.Span(id="explore-notice-text"),
                    html.Button("Retry", id="retry-data", className="btn btn-small", n_clicks=0),
                ]),
                html.Div(className="caption-row", children=[
                    html.Span(id="explore-caption", className="caption", children="Daily bikes hired vs temperature"),
                    html.Span(id="day-tag", className="tag", n_clicks=0, style={"display": "none"}),
                ]),
                html.Div(className="years", children=[
                    html.Span("Years", className="years-label"),
                    dcc.RangeSlider(id="year-range", min=years[0], max=years[1], step=1, value=list(years),
                                    marks=marks, allowCross=False, updatemode="mouseup",
                                    tooltip={"placement": "bottom", "always_visible": False}),
                ]),
                dcc.Loading(delay_show=400, overlay_style={"visibility": "visible", "opacity": 0.5},
                            custom_spinner=html.Div("Drawing…", className="loading-note"), children=[
                    dcc.Graph(id="scatter", style={"height": "520px"}, config={"displayModeBar": False, "responsive": True},
                              figure=empty_figure("Loading the bikes dataset…")),
                ]),
                html.Div(id="explore-insight", className="insight"),
                html.Div(id="explore-status", className="status"),
            ]),
            html.Div(className="side", children=[
                html.Div("Average hires by weekday", className="kicker"),
                html.Div(className="dow-list", children=weekday_rows()),
                html.Div(className="rule", children=[
                    html.Div(id="stat-label", className="kicker muted", children="All days"),
                    html.Div(className="stats", children=[
                        stat("stat-mean", "mean hires"),
                        stat("stat-corr", "r with temperature", label_id="stat-corr-label"),
                        stat("stat-max", "busiest day", sub_id="stat-max-sub"),
                        stat("stat-n", "days shown"),
                    ]),
                ]),
            ]),
        ]),
    ])


def slider_field(term, label, lo, hi, step, unit, _clamp) -> html.Div:
    in_model = term in COEFS
    return html.Div(className="field", children=[
        html.Label(children=[
            html.Span(label if in_model else f"{label} (not in model)"),
            html.B(id=f"d-{term}-label", children=f"+0 {unit}"),
        ]),
        dcc.Slider(id=f"d-{term}", min=lo, max=hi, step=step, value=0, marks=None, disabled=not in_model,
                   updatemode="mouseup", tooltip={"placement": "bottom", "always_visible": False}),
    ])


def coefficient_table() -> html.Div:
    if COEF_ERROR:
        return html.Div(className="notice", children=f"model_coefficients.csv could not be read: {COEF_ERROR}")
    rows = []
    for term in NUMERIC_TERMS:
        value = COEFS[term]
        rows.append(html.Tr([
            html.Td(term_label(term), title=term),
            html.Td(("+" if value >= 0 else "−") + f"{abs(value):,.0f}{term_unit(term)}",
                    className="pos" if value >= 0 else "neg"),
        ]))
    day_rows = []
    for d in DAYS:
        value = DAY_COEFS[d]
        text = "baseline" if d == "Mon" and value == 0 else signed(value)
        day_rows.append(html.Tr([html.Td(DAY_FULL[d]), html.Td(text, className="pos" if value >= 0 else "neg")]))
    return html.Div([
        html.Table(className="table", children=html.Tbody(rows)),
        html.Div("Weekday effect", className="kicker muted", style={"marginTop": "14px", "marginBottom": "8px"}),
        html.Table(className="table", children=html.Tbody(day_rows)),
        html.Div(f"Intercept {fmt(COEFS.get('Intercept'))}", className="status", style={"marginTop": "8px"}),
    ])


def predict_section() -> html.Div:
    return html.Section(id="section-predict", className="section", children=[
        html.Div(className="head", children=[
            html.H3("Predict"),
            html.Span("Intercept + Σ coefficient × weather + weekday effect", className="caption formula"),
            dcc.RadioItems(id="view-mode", className="seg", inline=True, value="cards",
                           options=[{"label": "Cards", "value": "cards"}, {"label": "Table", "value": "table"}]),
            html.Span(id="weather-status", className="status"),
            html.Button("Refresh weather", id="refresh-btn", className="btn btn-small", n_clicks=0,
                        title="Fetch the latest Open-Meteo data"),
        ]),
        html.Div(className="body", children=[
            dcc.Loading(delay_show=250, overlay_style={"visibility": "visible", "opacity": 0.45},
                        custom_spinner=html.Div("Fetching weather from Open-Meteo…", className="loading-note"),
                        parent_className="main main-loading", children=[
                html.Div(id="weather-notice", className="notice", style={"display": "none"}),
                html.Div([
                    html.Div(className="block-head", children=[
                        html.H4("First week of January 2026"),
                        html.Span(className="caption", children=["week total ", html.B(id="jan-total", children="–")]),
                    ]),
                    html.Div(id="jan-cards", className="cards cards-7"),
                ]),
                html.Div([
                    html.Div(className="block-head", children=[
                        html.H4(id="fc-heading", children="Next five days"),
                        html.Span(className="caption nowrap", children=[
                            "total ", html.B(id="fc-total", children="–"), " ",
                            html.Span(id="fc-delta", style={"color": N700}),
                        ]),
                    ]),
                    html.Div(className="fc-controls", children=[
                        html.Span("Forecast for", className="caption"),
                        dcc.Input(id="fc-location", type="text", value=DEFAULT_LOCATION, debounce=True,
                                  className="input", placeholder="City, e.g. Paris"),
                        dcc.RadioItems(id="fc-days", className="seg", inline=True, value=FORECAST_DAYS,
                                       options=[{"label": f"{n} days", "value": n} for n in (3, 5, 7)]),
                    ]),
                    html.Div(id="fc-cards", className="cards cards-5"),
                ]),
                html.Div(className="compare", children=[
                    html.Div("Both weeks by weekday · ink = January, red = forecast", className="caption"),
                    dcc.Graph(id="compare", style={"height": "230px"}, config={"displayModeBar": False, "responsive": True},
                              figure=empty_figure("")),
                ]),
            ]),
            html.Div(className="side side-predict", children=[
                html.Div(className="whatif-head", children=[
                    html.Div("What-if on the forecast", className="kicker"),
                    html.Button("Reset", id="reset-whatif", className="btn btn-small", n_clicks=0),
                ]),
                *[slider_field(*spec) for spec in WHATIF],
                html.Div(id="whatif-box", className="box"),
                html.Div(className="rule", children=[
                    html.Div("Coefficients", className="kicker muted", style={"marginBottom": "8px"}),
                    coefficient_table(),
                ]),
            ]),
        ]),
    ])


def serve_layout():
    df, _ = load_bikes()
    return html.Div(className="shell", children=[
        dcc.Store(id="tab-store", data="explore"),
        dcc.Store(id="day-store", data=None),
        dcc.Store(id="totals-store"),
        dcc.Store(id="countup-sink"),
        rail(),
        html.Div(className="content", children=[explore_section(df), predict_section()]),
    ])


# --------------------------------------------------------------------------
# Page template: Modernist styling + a little JS (keyboard shortcuts, count-up)
# --------------------------------------------------------------------------
INDEX_STRING = """<!DOCTYPE html>
<html lang="en">
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
:root{--bg:#f3f2f2;--surface:#eae9e9;--ink:#201e1d;--accent:#ec3013;--divider:rgba(32,30,29,.4);
--n300:#d7d3d3;--n400:#bab6b6;--n500:#9b9797;--n700:#605d5d;--n800:#444141;--font:"Archivo",system-ui,sans-serif}
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:15px;line-height:1.55}
h3,h4{margin:0;font-weight:800;letter-spacing:-.015em;line-height:1.12}
h4{font-size:20px}
b{font-weight:800}
.shell{display:grid;grid-template-columns:96px minmax(0,1fr);min-height:100vh;max-width:1440px;margin:0 auto}
.rail{background:var(--ink);color:var(--bg);display:flex;flex-direction:column;padding:20px 0;position:sticky;top:0;height:100vh}
.brand{width:20px;height:20px;background:var(--accent);margin:0 0 36px 20px;transition:transform .35s ease}
.brand:hover{transform:rotate(90deg)}
.rail-btn{background:transparent;color:var(--bg);border:0;border-left:4px solid transparent;text-align:left;
padding:14px 12px 14px 16px;font:800 12px/1.2 var(--font);cursor:pointer;transition:background .2s,color .2s,border-color .2s}
.rail-btn span{display:block;font-size:10px;opacity:.6;font-weight:400;margin-bottom:4px}
.rail-btn:hover{background:rgba(243,242,242,.08)}
.rail-btn.is-active{background:var(--bg);color:var(--ink);border-left-color:var(--accent)}
.rail-btn.is-active:hover{background:var(--bg)}
.rail-foot{margin-top:auto;padding:0 16px;font-size:10px;line-height:1.4;opacity:.55}
.content{min-width:0;display:flex;flex-direction:column}
.section{display:none;flex-direction:column;flex:1}
.section.is-active{display:flex;animation:fade .35s ease}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.head{display:flex;align-items:center;gap:16px;padding:16px 28px;border-bottom:2px solid var(--divider);flex-wrap:wrap}
.head h3{font-size:22px;margin-right:auto}
.body{display:grid;grid-template-columns:minmax(0,1fr) 340px;flex:1}
.main{padding:20px 28px;min-width:0;display:flex;flex-direction:column;gap:20px}
.side{border-left:2px solid var(--divider);padding:20px 24px;display:flex;flex-direction:column;gap:14px}
.side-predict{gap:18px}
.kicker{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.kicker.muted{color:var(--n700)}
.caption{font-size:13px;color:var(--n700)}
.caption b{color:var(--ink)}
.formula{font-size:12px}
.nowrap{white-space:nowrap;flex:none}
.status{font-size:11px;color:var(--n700)}
.caption-row{display:flex;align-items:baseline;gap:12px;margin-bottom:6px}
.insight{font-size:13px;line-height:1.5;padding:10px 12px;background:var(--surface);border-left:3px solid var(--ink)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;cursor:pointer;font:800 14px/1.2 var(--font);
color:var(--ink);background:transparent;border:1px solid var(--divider);padding:8px 14px;border-radius:0;
transition:background .15s,color .15s,border-color .15s}
.btn:hover{background:rgba(32,30,29,.07)}
.btn:active{background:rgba(32,30,29,.14)}
.btn.is-on{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.btn.is-on:hover{background:#dd2b0f}
.btn-small{font-size:12px;padding:5px 10px}
.btn-quiet{border-color:transparent;color:var(--n700)}
.btn-quiet:hover{color:var(--ink)}
.tag{display:inline-flex;align-items:center;font-size:11px;letter-spacing:.02em;padding:3px 10px;background:#fff2ef;color:#7c1405;cursor:pointer}
.tag:hover{background:#ffe0d9}
.rule{border-top:2px solid var(--divider);padding-top:14px;margin-top:6px}
.years{display:flex;align-items:center;gap:14px;margin:4px 0 8px}
.years-label{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--n700);flex:none}
.years > div{flex:1}
.dow-list{display:flex;flex-direction:column;gap:8px}
.dow{cursor:pointer;display:grid;grid-template-columns:36px minmax(0,1fr) 56px;gap:10px;align-items:center;font-size:13px;transition:color .2s}
.dow .track{height:18px;background:var(--n300)}
.dow .fill{height:18px;transition:width .45s cubic-bezier(.2,.7,.2,1),background .2s}
.dow:hover .track{box-shadow:inset 0 0 0 1px var(--divider)}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}
.stat-v{font-size:22px;font-weight:800;letter-spacing:-.02em}
.stat-l{font-size:11px;color:var(--n700)}
.stat-s{font-size:11px;color:var(--n500)}
.block-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.cards{display:grid;gap:2px;margin-top:10px;padding:1px}
.cards-7{grid-template-columns:repeat(7,minmax(0,1fr))}
.cards-5{grid-template-columns:repeat(auto-fit,minmax(0,1fr));grid-auto-flow:column}
.card{background:var(--surface);padding:12px;animation:fade .4s ease both;min-width:0;box-shadow:0 0 0 1px var(--divider)}
.card.fc{border-top:3px solid var(--accent)}
.card .d{font-size:11px;color:var(--n700)}
.card .p{font-size:20px;font-weight:800;letter-spacing:-.02em;margin:6px 0 4px;white-space:nowrap}
.card.fc .p{color:var(--accent)}
.card .w{font-size:11px;color:var(--n800);line-height:1.5}
.card .delta{font-size:11px;font-weight:800;margin-top:4px;color:var(--n700)}
.card.empty .p{color:var(--n400)}
.fc-controls{display:flex;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap}
.table-wrap{margin-top:10px;overflow-x:auto;animation:fade .35s ease}
.ptable{width:100%;border-collapse:collapse;font-size:12px;background:var(--surface)}
.ptable th{text-align:left;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--n700);padding:8px 7px;border-bottom:2px solid var(--divider);white-space:nowrap;cursor:default}
.ptable td{padding:6px 7px;border-bottom:1px solid var(--divider);white-space:nowrap}
.ptable td.num,.ptable th:nth-child(n+3){text-align:right}
.ptable td.pred{font-weight:800}
.ptable td.pred.fc{color:var(--accent)}
.ptable td.total{font-weight:800;border-bottom:0}
.ptable caption{caption-side:bottom;text-align:left;font-size:11px;color:var(--n700);padding:6px 0 0}
.compare{flex:1;min-height:0}
.whatif-head{display:flex;justify-content:space-between;align-items:center}
.field label{display:flex;justify-content:space-between;font-size:12px;color:rgba(32,30,29,.7);margin-bottom:6px}
.field label b{color:var(--ink)}
.box{font-size:13px;line-height:1.5;padding:12px;background:var(--surface)}
.box .fun{color:var(--n700);display:block;margin-top:6px;font-size:12px}
.notice{font-size:12px;color:var(--n800);padding:8px 12px;background:var(--surface);border-left:3px solid var(--accent);
display:flex;align-items:center;gap:12px;flex-wrap:wrap;line-height:1.5}
.notice span{flex:1;min-width:200px}
.table{width:100%;border-collapse:collapse;font-size:12px}
.table td{padding:5px 0;border-bottom:1px solid var(--divider)}
.table td:last-child{text-align:right;font-weight:800}
.pos{color:var(--accent)}.neg{color:var(--ink)}
.loading-note{font:800 12px var(--font);color:var(--ink);letter-spacing:.06em;text-transform:uppercase;padding:8px 12px;background:var(--bg);border:1px solid var(--divider)}
.input{min-height:36px;padding:6px 10px;font:inherit;font-size:14px;color:var(--ink);background:var(--surface);
border:1px solid var(--divider);border-radius:0;width:170px;caret-color:var(--accent)}
.input:hover{border-color:rgba(32,30,29,.6)}
.input:focus{outline:none;border-color:var(--accent)}
/* segmented radio (dcc.RadioItems) */
.seg{display:inline-flex;border:1px solid var(--divider);background:var(--bg)}
.seg label,.seg .dash-radioitems-inline{display:inline-flex;align-items:center;gap:6px;padding:7px 12px;font-size:13px;cursor:pointer;margin:0;transition:background .15s,color .15s}
.seg > *+*{border-left:1px solid var(--divider)}
.seg input{position:absolute;opacity:0;width:0;height:0;margin:0;pointer-events:none}
.seg label:has(input:checked),.seg .dash-radioitems-inline:has(input:checked){background:var(--accent);color:var(--bg)}
.seg label:not(:has(input:checked)):hover{background:rgba(32,30,29,.07)}
/* dcc.Dropdown (Dash 4) */
.dd .dash-dropdown-trigger,.dd .dash-dropdown{border-radius:0!important;font-family:var(--font)!important;font-size:14px!important}
.dd .dash-dropdown-trigger{background:var(--surface)!important;border:1px solid var(--divider)!important;min-height:36px!important;color:var(--ink)!important;box-shadow:none!important}
.dd .dash-dropdown-trigger:hover{border-color:rgba(32,30,29,.6)!important}
.dd .dash-dropdown-value{color:var(--ink)!important}
.dash-dropdown-content{border-radius:0!important;border:1px solid var(--divider)!important;background:var(--bg)!important;font-family:var(--font)!important;box-shadow:0 3px 10px rgba(45,43,43,.16)!important}
.dash-dropdown-option{font-size:13px!important;border-radius:0!important;color:var(--ink)!important}
.dash-dropdown-option:hover,.dash-dropdown-option[data-highlighted],.dash-dropdown-option[aria-selected="true"]{background:var(--accent)!important;color:var(--bg)!important}
/* dcc.Slider / RangeSlider (Dash 4) */
.dash-slider-track{background:var(--n300)!important;height:2px!important;border-radius:0!important}
.dash-slider-range{background:var(--accent)!important;border-radius:0!important}
.dash-slider-thumb{width:14px!important;height:14px!important;border-radius:0!important;background:var(--accent)!important;border:0!important;box-shadow:none!important;cursor:pointer}
.dash-slider-thumb:focus-visible{outline:2px solid var(--ink)!important;outline-offset:2px}
.dash-slider-mark{font-size:10px!important;color:var(--n700)!important;font-family:var(--font)!important}
.dash-slider-dot{border-radius:0!important;border-color:var(--n400)!important}
.dash-slider-tooltip{font-family:var(--font)!important;font-size:11px!important;background:var(--ink)!important;color:var(--bg)!important;border-radius:0!important}
.field .dash-slider-thumb{background:var(--accent)!important}
@media (max-width:1100px){.body{grid-template-columns:1fr}.side{border-left:0;border-top:2px solid var(--divider)}.cards-7{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media (max-width:700px){.shell{grid-template-columns:64px minmax(0,1fr)}.rail-btn{padding:12px 6px 12px 10px}.cards-7,.cards-5{grid-template-columns:repeat(2,minmax(0,1fr));grid-auto-flow:row}.rail-foot{display:none}}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
<script>
document.addEventListener('keydown', function (e) {
  var t = e.target && e.target.tagName;
  if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) { return; }
  if (e.key === '1') { var a = document.getElementById('rail-explore'); if (a) { a.click(); } }
  if (e.key === '2') { var b = document.getElementById('rail-predict'); if (b) { b.click(); } }
});
window.bikesCountUp = function (id, target) {
  var el = document.getElementById(id);
  if (!el || target === null || target === undefined) { return; }
  var start = parseFloat(el.dataset.v || '0') || 0;
  if (el._raf) { cancelAnimationFrame(el._raf); }
  if (start === target) { el.textContent = Math.round(target).toLocaleString('en-GB'); return; }
  var t0 = performance.now(), dur = 650;
  function step(now) {
    var k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = Math.round(start + (target - start) * e).toLocaleString('en-GB');
    if (k < 1) { el._raf = requestAnimationFrame(step); } else { el.dataset.v = String(target); el._raf = null; }
  }
  el._raf = requestAnimationFrame(step);
};
</script>
</body>
</html>"""


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = Dash(
    __name__,
    title="TfL bikes · Explore & Predict",
    update_title=None,
    external_stylesheets=[FONT_URL],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.index_string = INDEX_STRING
app.layout = serve_layout
server = app.server  # for gunicorn: `gunicorn app:server`


# --- section switching (client side, instant) ----------------------------
clientside_callback(
    """
    function (nExplore, nPredict, current) {
        var trig = (window.dash_clientside.callback_context.triggered || []);
        var tab = current || 'explore';
        if (trig.length && trig[0].value) {
            tab = trig[0].prop_id.indexOf('rail-predict') === 0 ? 'predict' : 'explore';
        }
        var ex = tab === 'explore';
        return [tab, ex ? 'section is-active' : 'section', ex ? 'section' : 'section is-active',
                ex ? 'rail-btn is-active' : 'rail-btn', ex ? 'rail-btn' : 'rail-btn is-active'];
    }
    """,
    Output("tab-store", "data"),
    Output("section-explore", "className"),
    Output("section-predict", "className"),
    Output("rail-explore", "className"),
    Output("rail-predict", "className"),
    Input("rail-explore", "n_clicks"),
    Input("rail-predict", "n_clicks"),
    State("tab-store", "data"),
)

# --- what-if slider labels follow the thumb while dragging -----------------
clientside_callback(
    """
    function (dT, dP, dW, vT, vP, vW) {
        var trig = (window.dash_clientside.callback_context.triggered || []).map(function (t) { return t.prop_id; });
        function pick(id, drag, val) {
            var v = trig.indexOf(id + '.value') >= 0 || drag === null || drag === undefined ? val : drag;
            return v === null || v === undefined ? 0 : v;
        }
        function label(v, unit) {
            var n = Math.abs(v), txt = Number.isInteger(n) ? String(n) : n.toFixed(1);
            return (v >= 0 ? '+' : '−') + txt + ' ' + unit;
        }
        return [label(pick('d-temp', dT, vT), '°C'), label(pick('d-precip', dP, vP), 'mm'),
                label(pick('d-windspeed', dW, vW), 'km/h')];
    }
    """,
    Output("d-temp-label", "children"),
    Output("d-precip-label", "children"),
    Output("d-windspeed-label", "children"),
    Input("d-temp", "drag_value"),
    Input("d-precip", "drag_value"),
    Input("d-windspeed", "drag_value"),
    Input("d-temp", "value"),
    Input("d-precip", "value"),
    Input("d-windspeed", "value"),
)

# --- animated totals -------------------------------------------------------
clientside_callback(
    """
    function (totals) {
        if (!totals || !window.bikesCountUp) { return window.dash_clientside.no_update; }
        Object.keys(totals).forEach(function (id) { window.bikesCountUp(id, totals[id]); });
        return window.dash_clientside.no_update;
    }
    """,
    Output("countup-sink", "data"),
    Input("totals-store", "data"),
)


@app.callback(
    Output("day-store", "data"),
    Input({"type": "dow-row", "day": ALL}, "n_clicks"),
    Input("day-tag", "n_clicks"),
    Input("shuffle-btn", "n_clicks"),
    State("day-store", "data"),
    prevent_initial_call=True,
)
def toggle_day(_rows, _tag, _shuffle, current):
    """Weekday row click toggles the filter; the ✕ tag clears it; shuffle picks at random."""
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update
    tid = ctx.triggered_id
    if tid == "day-tag":
        return None
    if tid == "shuffle-btn":
        return random.choice([None, None, *DAYS])
    if isinstance(tid, dict) and tid.get("type") == "dow-row":
        day = tid.get("day")
        return None if day == current else day
    return no_update


@app.callback(
    Output("weather-var", "value"),
    Output("color-by", "value"),
    Output("year-range", "value"),
    Input("shuffle-btn", "n_clicks"),
    State("year-range", "min"),
    State("year-range", "max"),
    prevent_initial_call=True,
)
def shuffle(n_clicks, lo, hi):
    if not n_clicks:
        return no_update, no_update, no_update
    lo, hi = int(lo), int(hi)
    span = random.choice([hi - lo, hi - lo, 1, 2, 3, 5, 8])
    start = random.randint(lo, max(lo, hi - span))
    return random.choice(list(VARS)), random.choice(["weekend", "season"]), [start, min(hi, start + span)]


@app.callback(
    Output("d-temp", "value"),
    Output("d-precip", "value"),
    Output("d-windspeed", "value"),
    Input("reset-whatif", "n_clicks"),
    prevent_initial_call=True,
)
def reset_whatif(n_clicks):
    if not n_clicks:
        return no_update, no_update, no_update
    return 0, 0, 0


@app.callback(
    Output("scatter", "figure"),
    Output("explore-caption", "children"),
    Output("day-tag", "children"),
    Output("day-tag", "style"),
    Output("trend-btn", "className"),
    Output({"type": "dow-fill", "day": ALL}, "style"),
    Output({"type": "dow-val", "day": ALL}, "children"),
    Output({"type": "dow-row", "day": ALL}, "style"),
    Output("stat-label", "children"),
    Output("stat-mean", "children"),
    Output("stat-corr", "children"),
    Output("stat-corr-label", "children"),
    Output("stat-max", "children"),
    Output("stat-max-sub", "children"),
    Output("stat-n", "children"),
    Output("explore-insight", "children"),
    Output("explore-notice-text", "children"),
    Output("explore-notice", "style"),
    Output("explore-status", "children"),
    Input("weather-var", "value"),
    Input("color-by", "value"),
    Input("trend-btn", "n_clicks"),
    Input("day-store", "data"),
    Input("year-range", "value"),
    Input("retry-data", "n_clicks"),
)
def update_explore(var, color_by, trend_clicks, day, years, _retry):
    var = var if var in VARS else "temp"
    label, short, unit = VARS[var]
    trend_on = bool((trend_clicks or 0) % 2)
    trend_class = "btn is-on" if trend_on else "btn"
    force = triggered_id() == "retry-data"
    df, error = load_bikes(force=force)

    if df is None:
        fills = [{"width": "0%", "background": INK} for _ in DAYS]
        vals = ["–"] * 7
        row_styles = [{"color": INK, "fontWeight": 400} for _ in DAYS]
        notice = f"The bikes dataset could not be loaded ({error or 'unknown error'}). The charts will fill in once it is available."
        return (empty_figure("Bikes dataset unavailable"), f"Daily bikes hired vs {short}", "", {"display": "none"},
                trend_class, fills, vals, row_styles, "All days", "–", "–", f"r with {short}", "–", "", "–", "",
                notice, {"display": "flex"}, "")

    lo, hi = (int(years[0]), int(years[1])) if years and len(years) == 2 else (int(df["year"].min()), int(df["year"].max()))
    in_years = df[(df["year"] >= lo) & (df["year"] <= hi)]
    rows = in_years[in_years["day_of_week"] == day] if day in DAYS else in_years

    # weekday panel (year filter applies, weekday filter only highlights)
    avg = in_years.groupby("day_of_week")["bikes_hired"].mean()
    max_avg = max(1.0, float(avg.max())) if len(avg) else 1.0
    fills, vals, row_styles = [], [], []
    for d in DAYS:
        a = float(avg.get(d, 0.0)) if d in avg.index else 0.0
        if day == d:
            fill, ink, weight = RED, INK, 800
        elif day in DAYS:
            fill, ink, weight = N400, N500, 400
        else:
            fill, ink, weight = INK, INK, 400
        fills.append({"width": f"{a / max_avg * 100:.1f}%", "background": fill})
        vals.append(fmt(a) if a else "–")
        row_styles.append({"color": ink, "fontWeight": weight})

    # stats over the rows shown
    n = int(len(rows))
    mean = float(rows["bikes_hired"].mean()) if n else float("nan")
    pair = rows[[var, "bikes_hired"]].dropna()
    corr = float(np.corrcoef(pair[var], pair["bikes_hired"])[0, 1]) if len(pair) > 2 and np.ptp(pair[var].to_numpy()) > 0 else float("nan")
    corr_text = f"{corr:.2f}".replace("-", "−") if not np.isnan(corr) else "–"
    if n:
        top = rows.loc[rows["bikes_hired"].idxmax()]
        max_text, max_sub = fmt(top["bikes_hired"]), str(top["label"])
    else:
        max_text, max_sub = "–", ""
    day_label = DAY_FULL[day] if day in DAYS else "All days"
    span = f"{lo}–{hi}" if (lo, hi) != (int(df["year"].min()), int(df["year"].max())) else ""
    caption = f"Daily bikes hired vs {short}" + (f" · {span}" if span else "")
    tag_children = f"{day_label} only ✕" if day in DAYS else ""
    tag_style = {"display": "inline-flex"} if day in DAYS else {"display": "none"}

    line = trend_line(rows, var)
    if n and line:
        best_day = avg.idxmax() if len(avg) else None
        insight = (f"Across {n:,} {day_label if day in DAYS else 'days'}"
                   f"{' in ' + span if span else ''}, each extra {unit} of {short} goes with "
                   f"{signed(line[0])} hires (r = {corr_text}).")
        if best_day and day not in DAYS:
            insight += f" {DAY_FULL[best_day]} are the busiest on average."
    elif n:
        insight = f"{n:,} {day_label if day in DAYS else 'days'} shown."
    else:
        insight = "No days match this selection. Widen the year range or clear the weekday filter."

    status = (f"{len(df):,} days · {df['date'].min():%-d %b %Y} – {df['date'].max():%-d %b %Y} · "
              f"source: {_bikes.get('source') or 'GitHub'}" + (" · drag on the chart to zoom, double-click to reset" if n else ""))
    fig = scatter_figure(rows, var, color_by, trend_on)
    return (fig, caption, tag_children, tag_style, trend_class, fills, vals, row_styles, day_label,
            fmt(mean), corr_text, f"r with {short}", max_text, max_sub, f"{n:,}", insight, "", {"display": "none"}, status)


def weather_card(row, forecast: bool, index: int, adjusted: bool = False, base_pred=None) -> html.Div:
    date = pd.Timestamp(row["date"])
    dow = row["day_of_week"]

    def shown(term):
        return row.get(f"used_{term}", row.get(term))

    temp, precip, wind = shown("temp"), shown("precip"), shown("windspeed")
    lines = []
    if temp is not None and not pd.isna(temp):
        lines.append(f"{temp:.1f}°C")
    if precip is not None and not pd.isna(precip):
        lines.append(f"{precip:.1f} mm")
    if wind is not None and not pd.isna(wind):
        lines.append(f"{wind:.0f} km/h")
    weather_lines = []
    for i, text in enumerate(lines):
        if i:
            weather_lines.append(html.Br())
        weather_lines.append(text)

    parts = [f"Intercept {fmt(COEFS.get('Intercept'))}"]
    for term in NUMERIC_TERMS:
        used = row.get(f"used_{term}")
        if used is not None and not pd.isna(used):
            parts.append(f"{term_label(term)} {used:,.1f} × {COEFS[term]:,.1f} = {signed(row[f'c_{term}'])}")
    parts.append(f"{DAY_FULL[dow] if dow in DAY_FULL else dow} {signed(row['day_effect'])}")
    if row["pred_raw"] < 0:
        parts.append(f"raw {row['pred_raw']:,.0f}, shown as 0")
    tooltip = " · ".join(parts)

    kids = [
        html.Div(f"{dow} · {date.day} {date:%b}", className="d"),
        html.Div(fmt(row["pred"]), className="p", title=tooltip),
        html.Div(weather_lines, className="w"),
    ]
    if adjusted and base_pred is not None:
        kids.append(html.Div(f"{signed(row['pred'] - base_pred)} what-if", className="delta"))
    return html.Div(kids, className="card fc" if forecast else "card", style={"animationDelay": f"{index * 45}ms"})


def placeholder_cards(dates, forecast: bool) -> list:
    return [html.Div([
        html.Div(f"{d:%a} · {d.day} {d:%b}", className="d"),
        html.Div("–", className="p"),
        html.Div("no weather", className="w"),
    ], className=("card fc empty" if forecast else "card empty"), style={"animationDelay": f"{i * 45}ms"})
        for i, d in enumerate(dates)]


TABLE_COLUMNS = [  # column -> (short header, full name, format)
    ("temp", ("Temp", "Temperature (°C)", "{:.1f}")), ("humidity", ("Hum.", "Humidity (%)", "{:.0f}")),
    ("precip", ("Rain", "Precipitation (mm)", "{:.1f}")), ("windspeed", ("Wind", "Wind speed (km/h)", "{:.0f}")),
    ("cloudcover", ("Cloud", "Cloud cover (%)", "{:.0f}")),
]


def prediction_table(pred: pd.DataFrame, forecast: bool, adjusted: bool = False, base: pd.DataFrame | None = None,
                     imputed: list | None = None) -> html.Table:
    """The same predictions as the cards, as one row per day (the assignment's 'table')."""
    imputed = [t for t in (imputed or []) if f"used_{t}" in pred.columns]
    head = [("Date", None), ("Day", None)]
    cols = []
    for col, (header, full, form) in TABLE_COLUMNS:
        source = f"used_{col}" if f"used_{col}" in pred.columns else (col if col in pred.columns else None)
        if source:
            head.append((header, full))
            cols.append((source, form))
    for term in imputed:
        short = term_label(term).replace("Precip. cover", "P. cover").replace("Solar radiation", "Solar").replace("Visibility", "Vis.")
        head.append((f"{short}*", f"{term_label(term)} ({term_unit(term)[1:]}), filled from the training data's monthly average"))
        cols.append((f"used_{term}", "{:.1f}"))
    head.append(("Day effect", "Weekday coefficient"))
    head.append(("Predicted", "Predicted bikes hired"))
    if adjusted:
        head.append(("What-if Δ", "Change versus the unadjusted forecast"))
    rows = []
    for i, r in pred.iterrows():
        cells = [html.Td(f"{pd.Timestamp(r['date']):%-d %b %Y}"), html.Td(r["day_of_week"])]
        for source, form in cols:
            value = r[source]
            cells.append(html.Td("–" if pd.isna(value) else form.format(float(value)), className="num"))
        cells.append(html.Td(signed(r["day_effect"]), className="num"))
        cells.append(html.Td(fmt(r["pred"]), className="num pred" + (" fc" if forecast else "")))
        if adjusted and base is not None:
            cells.append(html.Td(signed(r["pred"] - float(base["pred"].iloc[i])), className="num"))
        rows.append(html.Tr(cells))
    total = [html.Td("Total", colSpan=len(head) - 1 - (1 if adjusted else 0), className="total"),
             html.Td(fmt(pred["pred"].sum()), className="num pred total" + (" fc" if forecast else ""))]
    if adjusted and base is not None:
        total.append(html.Td(signed(pred["pred"].sum() - base["pred"].sum()), className="num total"))
    rows.append(html.Tr(total))
    caption = "* filled from the training data's monthly average" if imputed else None
    return html.Table(className="ptable", children=[
        html.Thead(html.Tr([html.Th(h, title=full) if full else html.Th(h) for h, full in head])),
        html.Tbody(rows),
        *( [html.Caption(caption)] if caption else [] ),
    ])


@app.callback(
    Output("jan-cards", "children"),
    Output("jan-cards", "className"),
    Output("jan-total", "children"),
    Output("fc-heading", "children"),
    Output("fc-cards", "children"),
    Output("fc-cards", "className"),
    Output("fc-total", "children"),
    Output("fc-delta", "children"),
    Output("fc-delta", "style"),
    Output("compare", "figure"),
    Output("whatif-box", "children"),
    Output("weather-notice", "children"),
    Output("weather-notice", "style"),
    Output("weather-status", "children"),
    Output("totals-store", "data"),
    Input("d-temp", "value"),
    Input("d-precip", "value"),
    Input("d-windspeed", "value"),
    Input("fc-location", "value"),
    Input("fc-days", "value"),
    Input("refresh-btn", "n_clicks"),
    Input("view-mode", "value"),
)
def update_predict(d_temp, d_precip, d_wind, location, days, _refresh, view_mode="cards"):
    force = triggered_id() == "refresh-btn"
    as_table = view_mode == "table"
    location = (location or DEFAULT_LOCATION).strip() or DEFAULT_LOCATION
    try:
        days = int(days) if days else FORECAST_DAYS
    except (TypeError, ValueError):
        days = FORECAST_DAYS
    days = min(7, max(1, days))
    adjust = {"temp": d_temp or 0, "precip": d_precip or 0, "windspeed": d_wind or 0}
    adjusted = any(adjust.values())

    archive = get_weather("archive", DEFAULT_LOCATION, force=force)
    forecast = get_weather("forecast", location, days, force=force)

    notes, imputed_all, missing_all = [], set(), set()
    if COEF_ERROR:
        notes.append(f"model_coefficients.csv could not be read ({COEF_ERROR}); predictions are unavailable.")
    if not COEFS:
        jan_pred = fc_pred = fc_base = None
    else:
        jan_pred = fc_pred = fc_base = None
        if archive["df"] is not None:
            jan_pred, imp, mis = predict(archive["df"])
            imputed_all.update(imp)
            missing_all.update(mis)
        if forecast["df"] is not None:
            fc_base, imp, mis = predict(forecast["df"])
            fc_pred, _, _ = predict(forecast["df"], adjust) if adjusted else (fc_base, imp, mis)
            imputed_all.update(imp)
            missing_all.update(mis)
    if archive["error"]:
        notes.append(f"Open-Meteo archive request for January 2026 failed ({archive['error']}).")
    if forecast["error"]:
        notes.append(f"Open-Meteo forecast request for {location} failed ({forecast['error']}).")
    if imputed_all:
        names = ", ".join(term_label(t).lower() for t in NUMERIC_TERMS if t in imputed_all)
        notes.append(f"Open-Meteo does not report {names}; the training data's average for that calendar month is used instead.")
    if missing_all:
        names = ", ".join(NUMERIC_TERMS_LABELS(t) for t in NUMERIC_TERMS if t in missing_all)
        notes.append(f"No source for {names}; those terms are set to zero.")
    if (archive["error"] or forecast["error"]) and not COEF_ERROR:
        notes.append("Use “Refresh weather” to try again.")

    # January cards (or table)
    imputed_list = [t for t in NUMERIC_TERMS if t in imputed_all]
    if jan_pred is not None:
        jan_cards = (prediction_table(jan_pred, False, imputed=imputed_list) if as_table
                     else [weather_card(r, False, i) for i, r in jan_pred.iterrows()])
        jan_total = float(jan_pred["pred"].sum())
        jan_avg = jan_total / max(1, len(jan_pred))
    else:
        jan_dates = pd.date_range(JAN_START, JAN_END)
        jan_cards = placeholder_cards(jan_dates, False)
        jan_total, jan_avg = None, None
    jan_class = "table-wrap" if (as_table and jan_pred is not None) else "cards cards-7"

    # forecast cards
    heading = f"Next {WORDS.get(days, days)} days"
    if forecast["df"] is not None and forecast.get("location") and forecast["location"].lower().split(",")[0] != DEFAULT_LOCATION.lower():
        heading += f" · {forecast['location']}"
    if fc_pred is not None:
        base_by_index = fc_base["pred"].tolist()
        fc_cards = (prediction_table(fc_pred, True, adjusted, fc_base, imputed_list) if as_table
                    else [weather_card(r, True, i, adjusted, base_by_index[i]) for i, r in fc_pred.iterrows()])
        fc_total = float(fc_pred["pred"].sum())
        fc_base_total = float(fc_base["pred"].sum())
        fc_avg = fc_total / max(1, len(fc_pred))
    else:
        today = datetime.now().date()
        fc_cards = placeholder_cards([pd.Timestamp(today + timedelta(days=i)) for i in range(days)], True)
        fc_total = fc_base_total = fc_avg = None
    fc_class = "table-wrap" if (as_table and fc_pred is not None) else "cards cards-5"

    if fc_avg is not None and jan_avg:
        delta = (fc_avg - jan_avg) / jan_avg
        delta_text = f"{'+' if delta >= 0 else '−'}{abs(delta) * 100:.0f}%/day vs Jan"
        delta_style = {"color": RED if delta >= 0 else N700}
    else:
        delta_text, delta_style = "", {"color": N700}

    if fc_total is not None:
        diff = fc_total - fc_base_total
        per_minute = fc_total / (len(fc_pred) * 24 * 60)
        box = [html.B(f"{signed(diff)} hires over {WORDS.get(len(fc_pred), len(fc_pred))} days"), html.Br(),
               html.Span("vs the unadjusted forecast total", style={"color": N700}) if adjusted
               else html.Span("move the sliders to see the model's sensitivity", style={"color": N700}),
               html.Span(f"≈ {per_minute:,.0f} bikes leaving a dock every minute, {fmt(fc_avg)} a day", className="fun")]
    else:
        box = [html.B("No forecast to adjust"), html.Br(),
               html.Span("the what-if sliders apply once Open-Meteo answers", style={"color": N700})]

    when = datetime.now().strftime("%H:%M")
    fetched = [k for k, hit in (("archive", archive), ("forecast", forecast)) if hit["df"] is not None]
    status = (f"Open-Meteo · {forecast.get('location') or location} · {' + '.join(fetched)} updated {when}"
              if fetched else "Open-Meteo · no data yet")
    totals = {"jan-total": jan_total, "fc-total": fc_total}
    return (jan_cards, jan_class, fmt(jan_total), heading, fc_cards, fc_class, fmt(fc_total), delta_text, delta_style,
            compare_figure(jan_pred, fc_pred), box, [html.Span(n) for n in notes],
            {"display": "flex"} if notes else {"display": "none"}, status, totals)


def NUMERIC_TERMS_LABELS(term: str) -> str:
    return term_label(term).lower()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DASH_DEBUG", "") == "1")

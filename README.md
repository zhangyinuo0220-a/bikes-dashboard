# TfL London bikes dashboard

A two-section [Dash](https://dash.plotly.com) app for the London cycle-hire scheme.

- **01 Explore** – scatter daily `bikes_hired` against a weather variable (temperature, humidity,
  precipitation, wind, cloud), colour by weekend or season, toggle an OLS trend line, narrow the years,
  and click a weekday in the "average hires by weekday" panel to filter everything to that day.
- **02 Predict** – applies the linear model in `model_coefficients.csv` to Open-Meteo weather:
  the first week of January 2026 (historical archive) and the next five days (live forecast),
  with what-if sliders on the forecast and a weekday comparison chart.

## Run locally

```bash
uv sync
uv run python app.py        # http://127.0.0.1:8050
```

`uv run gunicorn app:server` runs it the way Render does (gunicorn binds to `$PORT`).

## Deploy on Render

Push this folder to a public GitHub repo, then on Render choose **New > Web Service**, connect the repo and set

| Setting        | Value                          |
| -------------- | ------------------------------ |
| Build command  | `pip install uv && uv sync`    |
| Start command  | `uv run gunicorn app:server`   |

or use **New > Blueprint**, which reads `render.yaml`. No environment variables are required;
Render sets `PORT` itself and `.python-version` pins Python 3.12.

## Files

| File                     | Purpose                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| `app.py`                 | the whole Dash app (layout, callbacks, styling)                          |
| `open_meteo.py`          | `open_meteo()` forecast and `open_meteo_history()` archive helpers       |
| `model_coefficients.csv` | `term, coefficient` exported from the notebook (swap in the final model) |
| `data/london_bikes.csv`  | offline copy of the dataset, used only if the GitHub download fails      |
| `pyproject.toml`, `uv.lock` | dependencies (dash, plotly, pandas, requests, gunicorn)              |
| `render.yaml`            | Render blueprint with the build and start commands above                 |

## How predictions are made

`prediction = Intercept + Σ coefficient × value + day_<weekday>`, clamped at zero.
Terms are read from `model_coefficients.csv`, so a different set of numeric variables works without
code changes. `month` is taken from the date. Any numeric term that Open-Meteo does not provide
(for the current model: `visibility`, `solarradiation`, `precipcover`) is filled with the training
data's average for that calendar month, and the app says so in a notice on the Predict page.

Failed requests never break the page: the bikes dataset falls back to the bundled copy, and a failed
Open-Meteo call shows placeholder cards with a short notice and a **Refresh weather** button.

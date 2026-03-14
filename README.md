# StockOracle

StockOracle is a same-day stock mover ranking app built for intraday idea generation. It fetches market and intraday data, engineers bar-level and prior-day features, trains ensemble models on return-to-close targets, overlays live sentiment and options signals, and evaluates the output with a cost-aware backtest aligned to the current intraday decision slot.

## What it does

- Pulls OHLCV data with `yfinance`
- Pulls recent intraday bars and trains on same-bar-slot examples from prior sessions
- Builds momentum, volatility, volume, benchmark-relative, prior-day context, intraday bar state, and earnings-timing features
- Pulls live news sentiment and options chain structure for score overlays
- Trains an ensemble of tree models for:
  - same-day return-to-close prediction
  - same-day absolute move prediction
- Scores the latest tradable intraday bar with a composite rank
- Evaluates recent out-of-sample performance with an expanding-window holdout
- Simulates a top-k portfolio with transaction costs and slippage
- Exposes everything in both Streamlit and a Vercel-ready Next.js + Python API app

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Vercel deployment

This repo includes:

- a Next.js frontend in `app/`
- a Python API in `api/`
- `vercel.json` with Python function settings

Deploy steps:

```bash
npm install
vercel
```

Vercel will build the Next.js app and expose the Python endpoints:

- `/api`
- `/api/health`
- `/api/rank`

Production checklist:

- Confirm `npm run build` passes locally
- Confirm the Python API smoke test passes: `PYTHONPATH=src python -c "from api.rank import RankingRequest, rank; print(rank(RankingRequest(universe=['AAPL','MSFT','NVDA']))['ranking'][0]['symbol'])"`
- Set the Vercel framework preset to `Next.js`
- Keep the repo root as the Vercel root directory
- Verify Python functions are enabled for `api/*.py`
- If you want durable paper-trading state across deploys, point `STOCKORACLE_EXECUTION_DIR` to persistent storage instead of ephemeral serverless disk
- If you want shared cache persistence, point `STOCKORACLE_CACHE_DIR` to persistent storage

Operational notes:

- Yahoo-backed market-data requests are cached and rate-limited in-process to reduce throttling and repeated cold fetches
- The execution layer supports `paper` mode today through `/api/execute` and `/api/positions`
- Paper orders are generated from the current top-ranked names with capital and max-notional caps

## Local web app

Run the frontend locally:

```bash
npm install
npm run dev
```

The frontend posts ranking requests to `/api/rank`.

## Paper trading

The app includes a paper broker interface for same-day execution workflows.

Endpoints:

- `/api/rank` returns the ranking plus an execution plan
- `/api/execute` submits the current top picks to the paper broker
- `/api/positions` returns paper positions and recent orders

The broker layer is defined in `src/stockoracle/execution.py`. The current production implementation is `paper`, with a broker abstraction ready for a live adapter later.

## Notes

- The default universe is a liquid large-cap watchlist. You can replace it in the UI.
- The model ranks symbols by expected move from the current bar into the close. It does not guarantee returns.
- Live news and options inputs are current-state overlays, not point-in-time historical archives.
- Market regimes change. Treat this as a research tool, not a fully automated strategy.

## Project layout

```text
api/              Vercel Python API
app/              Next.js frontend
src/stockoracle/
  alternative_data.py  News, earnings, and options signals
  app.py          Core orchestration
  backtest.py     Cost-aware portfolio simulation
  config.py       Runtime configuration
  data.py         Market data fetching and normalization
  features.py     Feature engineering and targets
  modeling.py     Ensemble training, ranking, and evaluation
  universe.py     Default symbol universe
streamlit_app.py  Streamlit entrypoint
```
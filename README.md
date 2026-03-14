# StockOracle

StockOracle is a same-day stock mover ranking app built for intraday idea generation. It fetches Yahoo market and intraday data through direct HTTP APIs, engineers bar-level and prior-day features, trains ensemble models on return-to-close targets, overlays live sentiment and options signals, and evaluates the output with a cost-aware backtest aligned to the current intraday decision slot.

## What it does

- Pulls OHLCV data from Yahoo Finance chart APIs
- Pulls recent intraday bars and trains on same-bar-slot examples from prior sessions
- Builds momentum, volatility, volume, benchmark-relative, prior-day context, intraday bar state, and earnings-timing features
- Pulls live news sentiment and options chain structure for score overlays
- Trains a blended ensemble of tree and full-batch neural gradient-descent models for:
  - same-day return-to-close prediction
  - same-day absolute move prediction
- Estimates direction probability and return bands for long/short opportunity ranking
- Scores the latest tradable intraday bar with a composite rank
- Evaluates recent out-of-sample performance with an expanding-window holdout
- Simulates a top-k portfolio with transaction costs and slippage
- Exposes everything in both Streamlit and a Vercel-ready Next.js + Python API app

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt
streamlit run streamlit_app.py
```

For Vercel and Python API deployment, the root `requirements.txt` is intentionally trimmed to runtime-only dependencies so the Lambda bundle stays under the 500 MB ephemeral storage limit.
The deployment is pinned to Python 3.12 via `runtime.txt`, and the runtime requirements are pinned to versions with broadly available pre-built binary wheels for NumPy, pandas, SciPy, and scikit-learn.

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
- Keep `requirements.txt` limited to API runtime dependencies; use `requirements-local.txt` for the Streamlit research UI
- Keep Vercel on Python 3.12 and deploy from `requirements.txt` instead of a `uv.lock`/`pyproject.toml` flow
- Set `STOCKORACLE_SESSION_SECRET`, `STOCKORACLE_EXECUTION_TOKEN`, and `STOCKORACLE_CONFIRMATION_SECRET` in Vercel before enabling authenticated trade flows
- Use `STOCKORACLE_REDIS_URL` for durable shared storage across serverless instances, or `STOCKORACLE_STORAGE_DIR` only for local disk-backed development

Operational notes:

- Yahoo-backed market-data requests are cached and rate-limited in-process to reduce throttling and repeated cold fetches
- The execution layer supports `paper` and `alpaca` modes through `/api/execute` and `/api/positions`
- Paper orders are generated from the current top-ranked names with capital and max-notional caps

Environment variables:

- `STOCKORACLE_OPERATOR_USERNAME`: operator username for the web login, defaults to `operator`
- `STOCKORACLE_OPERATOR_PASSWORD`: operator password for protected trade actions
- `STOCKORACLE_SESSION_SECRET`: required secret used to sign web login sessions
- `STOCKORACLE_EXECUTION_TOKEN`: required token for `/api/execute` and `/api/positions`
- `STOCKORACLE_CONFIRMATION_SECRET`: required secret used to sign confirmation tokens for execution plans
- `STOCKORACLE_REDIS_URL`: enables durable Redis-backed storage for cache and paper broker state
- `STOCKORACLE_STORAGE_DIR`: optional local storage root for development when Redis is not used
- `ALPACA_API_KEY`: Alpaca key for live execution mode
- `ALPACA_SECRET_KEY`: Alpaca secret for live execution mode
- `ALPACA_BASE_URL`: optional Alpaca base URL, defaults to paper trading

## Local web app

Run the frontend locally:

```bash
npm install
npm run dev
```

The frontend posts ranking requests to `/api/rank`.

For LLMs, bots, or external parsers using the Vercel app directly, use the Next endpoint `/api/predictions`.

You can omit symbols entirely and let StockOracle discover current global movers automatically.

Examples:

```bash
curl "https://your-app.vercel.app/api/predictions?symbols=AAPL,MSFT,NVDA&topK=3"
```

```bash
curl -X POST "https://your-app.vercel.app/api/predictions" \
  -H "Content-Type: application/json" \
  -d '{"universe":["AAPL","MSFT","NVDA"],"topK":3,"intradayInterval":"15m"}'
```

```bash
curl -X POST "https://your-app.vercel.app/api/predictions" \
  -H "Content-Type: application/json" \
  -d '{"discoverGlobalMovers":true,"globalMoversLimit":60,"topK":5,"intradayInterval":"15m"}'
```

To fetch the discovered symbol set directly from the Next app, use `/api/universe/global-movers?limit=60`.

The response is parser-friendly JSON with:

- `generatedAt`
- `request`
- `summary`
- `predictions`
- `metrics`
- `executionPlan`

## Browser AI assistant

The main web UI now includes a browser-side WebLLM assistant that uses a Qwen3 model to interpret results and call `/api/predictions` based on the user's interests.

Notes:

- It runs fully in the browser and requires WebGPU support
- The default packaged model is `Qwen3-8B-q4f32_1-MLC`, which is the closest WebLLM browser build to a Qwen3 9B-class setup
- You can override the default model id with `NEXT_PUBLIC_WEBLLM_MODEL`
- The assistant can request a market-wide mover scan by calling `/api/predictions` with `discoverGlobalMovers=true`

## Paper trading

The app includes a paper broker interface for same-day execution workflows.

Endpoints:

- `/api/rank` returns the ranking plus an execution plan
- `/api/execute` submits the current top picks to the paper broker
- `/api/positions` returns paper positions and recent orders
- `/api/autopilot/run` executes the global daily autopilot cycle
- `/api/autopilot/close` flattens the global daily autopilot positions near the close
- `/api/autopilot/status` returns persistent global autopilot state

The broker layer is defined in `src/stockoracle/execution.py`. The current implementations are `paper` and `alpaca`.

Execution safety:

- The web UI now requires operator login before it can submit or inspect authenticated trade routes
- `/api/execute` and `/api/positions` fail closed unless `STOCKORACLE_EXECUTION_TOKEN` is configured
- The rank response returns a confirmation token derived from the current execution plan
- `/api/execute` requires an explicit confirmation flag plus a matching confirmation token signed with `STOCKORACLE_CONFIRMATION_SECRET`, so stale plans are rejected

## Global autopilot

The app can run a global server-managed trading cycle instead of relying on a browser session.

Set these environment variables in Vercel:

- `STOCKORACLE_AUTOPILOT_ENABLED=true`
- `STOCKORACLE_AUTOPILOT_MODE=paper` or `alpaca`
- `STOCKORACLE_AUTOPILOT_DAILY_BUDGET=10000`
- `STOCKORACLE_AUTOPILOT_TOKEN=<long-random-secret>` or configure `CRON_SECRET`
- `STOCKORACLE_AUTOPILOT_UNIVERSE=AAPL,MSFT,NVDA,AMZN,...`
- `STOCKORACLE_AUTOPILOT_TOP_K=4`

For scheduling on Vercel free plans, use the included GitHub Actions workflow instead of Vercel Cron.
Set these GitHub repository secrets:

- `STOCKORACLE_APP_URL=https://your-app.vercel.app`
- `STOCKORACLE_AUTOPILOT_TOKEN=<same token configured on Vercel>`

Behavior:

- GitHub Actions calls `/api/autopilot/run` every 10 minutes on weekdays and the endpoint only executes once inside the New York run window, default `15:45`
- GitHub Actions calls `/api/autopilot/close` every 5 minutes on weekdays and the endpoint only executes once inside the New York close window, default `15:58`
- The autopilot uses persistent global storage, so positions and run history are shared across users and deployments when `STOCKORACLE_REDIS_URL` is configured
- The daily budget defaults to `$10,000`
- The controller adapts `top_k` and position concentration based on recent closeout performance, so it can de-risk or concentrate automatically over time

Recommended deployment notes:

- Use `paper` mode first
- Configure `STOCKORACLE_REDIS_URL` so autopilot state survives serverless instance churn
- If you enable `alpaca`, also set `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and optionally `ALPACA_BASE_URL`
- Use a separate secret for `STOCKORACLE_AUTOPILOT_TOKEN` instead of reusing public UI credentials
- The workflow is in `.github/workflows/autopilot-scheduler.yml` and can also be triggered manually with `workflow_dispatch`

## Notes

- The default universe is a liquid large-cap watchlist. You can replace it in the UI.
- The UI can also switch into a global mover discovery mode that sources liquid names from Yahoo predefined market screeners.
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
  execution.py    Paper and Alpaca execution adapters
  features.py     Feature engineering and targets
  modeling.py     Ensemble training, ranking, and evaluation
  runtime.py      Cache and throttling helpers
  storage.py      Durable file/Redis storage backends
  universe.py     Default symbol universe and global mover discovery
streamlit_app.py  Streamlit entrypoint
```
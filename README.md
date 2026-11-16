# Hockeypicks AI — Automated NHL Prediction Pipeline & Dashboard

Automated CPU-based NHL machine learning pipeline and static GitHub Pages web app converted from the `hockey_playoff_predictor.ipynb` Colab workbook and `h0sk1/hockeypicks` Space.

## Architecture
- **`src/data_sync.py`** — Syncs MoneyPuck season summary CSVs (2021–present) with Chrome/Cloudflare headers, compiles leak-free NHL API team game logs + streaks (`api-web.nhle.com`), fetches daily/tomorrow schedules (`gameWeek`), live skater stats (`gameType` `2` Regular Season / `3` Playoffs auto-detected), and Tim Hortons Challenge picks.
- **`src/skater_model.py`** — Trains an `XGBRegressor` on dynamic per-game skater features (fixing rate-column summation & saved feature-order mismatches from V3.1), calculates Empirical Bayes `N_half`, and scores both Tim Hortons Challenge groups and all active slate skaters with 70/30 defense matchup multipliers.
- **`src/team_game_model.py`** — Trains a Pairwise `XGBClassifier` on historical `Home - Away` diffs (Top-20 team features + Top-3 Star Power + Goalie GSAx/SV% + Streak/Bounce-Back momentum, weighting playoffs 5x), optimizes Log-Loss Temperature scaling (`T`) via 5-fold CV, generates 32-team AI Power Rankings, predicts upcoming games & Best-of-7 playoff series, and optionally evaluates Quarter-Kelly betting edges when `ODDS_API_KEY` is configured.
- **`main_pipeline.py` & `src/report_builder.py`** — Split headless runner supporting `--mode train` (trains XGBoost models & exports `models/` snapshots), `--mode infer` (loads saved `models/` for fast live slate/Tims/odds inference), or `--mode full` (default), exporting `docs/data.json`, `docs/tims_history.json`, and `docs/latest_run.html`.
- **`docs/index.html`** — Zero-bounce mobile-hardened GitHub Pages dashboard (Skater Picks + Forward Tracker, Game Matchups, 32-Team Power Rankings, Model Calibration/Feature Drivers, Run Logs, and an in-app **⚡ Run Inference** button that triggers `hockeypicks_infer.yml`).

## Setup (One-Time on GitHub)
1. Enable **GitHub Pages**: go to **Settings → Pages → Build and deployment → Source: Deploy from a branch**, select **`main`** and folder **`/docs`**, and click **Save**.
2. *(Optional)* Enable live sharp odds & Quarter-Kelly calculations: go to **Settings → Secrets and variables → Actions → New repository secret**, add **`ODDS_API_KEY`**.
3. Go to **Actions → Hockeypicks Automated AI Pipeline → Run workflow** to trigger your first full training + inference run immediately (scheduled training runs execute daily at **10:17 AM ET**, **4:17 PM ET**, and **11:43 PM ET** year-round with EDT/EST DST validation).
4. *(Optional)* To run **⚡ Run Inference** (`hockeypicks_infer.yml`) directly from the web app, click **⚙️** in the dashboard header and save a GitHub Fine-Grained PAT scoped strictly to **Repository: `hockeypicks-auto` → Actions: Read and write** (stored only in your browser's `localStorage`).

## Local Usage
```bash
pip install -r requirements.txt
python main_pipeline.py --mode full   # train + infer
python main_pipeline.py --mode train  # training only -> models/
python main_pipeline.py --mode infer  # live inference only -> docs/
python -m unittest discover -s tests
```

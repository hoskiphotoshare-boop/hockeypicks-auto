# Hockeypicks AI — Automated NHL Prediction Pipeline & Dashboard

Automated CPU-based NHL machine learning pipeline and static GitHub Pages web app converted from the `hockey_playoff_predictor.ipynb` Colab workbook and `h0sk1/hockeypicks` Space.

## Architecture
- **`src/data_sync.py`** — Syncs MoneyPuck season summary CSVs (2021–present) with Chrome/Cloudflare headers, compiles leak-free NHL API team game logs + streaks (`api-web.nhle.com`), fetches daily/tomorrow schedules (`gameWeek`), live skater stats (`gameType` `2` Regular Season / `3` Playoffs auto-detected), and Tim Hortons Challenge picks.
- **`src/skater_model.py`** — Trains an `XGBRegressor` on dynamic per-game skater features (fixing rate-column summation & saved feature-order mismatches from V3.1), calculates Empirical Bayes `N_half`, and scores both Tim Hortons Challenge groups and all active slate skaters with 70/30 defense matchup multipliers.
- **`src/team_game_model.py`** — Trains a Pairwise `XGBClassifier` on historical `Home - Away` diffs (Top-20 team features + Top-3 Star Power + Goalie GSAx/SV% + Streak/Bounce-Back momentum, weighting playoffs 5x), optimizes Log-Loss Temperature scaling (`T`) via 5-fold CV, generates 32-team AI Power Rankings, predicts upcoming games & Best-of-7 playoff series, and optionally evaluates Quarter-Kelly betting edges when `ODDS_API_KEY` is configured.
- **`main_pipeline.py` & `src/report_builder.py`** — Split headless runner supporting `--mode train` (trains XGBoost models & exports `models/` snapshots), `--mode infer` (loads saved `models/` for fast live slate/Tims/odds inference), or `--mode full` (default), exporting `docs/data.json`, `docs/tims_history.json`, and `docs/latest_run.html`.
- **`src/infer_gate.py`** — Lightweight gate run by `hockeypicks_infer.yml` supporting dual-cron Eastern Prevailing Time (EPT) seasonal gating for:
  - **10:00 PM EPT (day before)**: Daily opening betting odds release & next-day line movements.
  - **11:00 AM EPT**: Weekend & holiday early matinee odds release & line movements (evaluated dynamically on Saturdays, Sundays, US/Canadian holidays, or any weekday with games scheduled before 4:00 PM ET; skips idle weekdays in ~0.5s).
  - **12:00 PM EPT**: Daily noon betting odds release & mid-day line movements.
  - **4:30 PM EPT**: Daily pre-game odds movements & starting goalie confirmations.
  - **Post-Puck-Drop Reactive Ticks**: `:03/:18/:33/:48` past the hour across NHL game hours (~10:00 AM ET - 12:48 AM ET), executing fast live inference ~3+ minutes after each scheduled game starts without re-running empty windows.
- **`docs/index.html`** — Zero-bounce mobile-hardened GitHub Pages dashboard (Skater Picks + Forward Tracker, Game Matchups, 32-Team Power Rankings, Model Calibration/Feature Drivers, and Run Logs).

## Setup (One-Time on GitHub)
1. Enable **GitHub Pages**: go to **Settings → Pages → Build and deployment → Source: Deploy from a branch**, select **`main`** and folder **`/docs`**, and click **Save**.
2. *(Optional)* Enable live sharp odds & Quarter-Kelly calculations: go to **Settings → Secrets and variables → Actions → New repository secret**, add **`ODDS_API_KEY`**.
3. Go to **Actions → Hockeypicks Automated AI Pipeline → Run workflow** to trigger your first full training + inference run immediately (scheduled training runs execute daily at **12:01 AM ET**, **10:17 AM ET**, and **4:17 PM ET** year-round with EDT/EST DST validation, while post-puck-drop inference runs automatically via `hockeypicks_infer.yml`).

## Local Usage
```bash
pip install -r requirements.txt
python main_pipeline.py --mode full   # train + infer
python main_pipeline.py --mode train  # training only -> models/
python main_pipeline.py --mode infer  # live inference only -> docs/
python -m unittest discover -s tests
```

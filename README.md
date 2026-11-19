# Hockeypicks AI — Automated NHL Prediction Pipeline & Dashboard

Automated CPU-based NHL machine learning pipeline, live in-game updating service, and static GitHub Pages web app converted from `hockey_playoff_predictor.ipynb` and `h0sk1/hockeypicks`.

---

## Architecture Overview

### Data, Modeling & Inference Core
- **`main_pipeline.py`** — Headless entry point supporting `--mode train` (trains models & exports `models/` snapshots), `--mode infer` (loads saved `models/` for fast slate, Tims & odds inference), and `--mode full` (default), writing outputs to `docs/data.json`, `docs/tims_history.json`, and `docs/latest_run.html`.
- **`src/data_sync.py`** — Syncs MoneyPuck season summary CSVs (2021–present) with Chrome/Cloudflare headers, compiles leak-free NHL API team game logs + streaks (`api-web.nhle.com`), fetches daily/tomorrow schedules (`gameWeek`), pulls live club skater stats (`gameType` `2` Regular Season / `3` Playoffs auto-detected), and retrieves Tim Hortons Challenge player lists.
- **`src/availability.py`** — Ingests live NHL injury reports (ESPN with CBS fallback) and Daily Faceoff starting goalie confirmations into team and skater availability adjustments.
- **`src/skater_model.py`** — Trains an `XGBRegressor` on dynamic per-game skater features, computes Empirical Bayes shrinkage (`N_half`), and scores both Tim Hortons Challenge groups and active slate skaters with 70/30 defense matchup multipliers.
- **`src/team_game_model.py`** — Trains a Pairwise `XGBClassifier` on historical `Home - Away` diffs (Top-20 team features + Top-3 Star Power + Goalie GSAx/SV% + Streak/Bounce-Back momentum, weighting playoffs 5x), calibrates Log-Loss Temperature (`T`) via 5-fold CV, generates 32-team AI Power Rankings, predicts upcoming games & Best-of-7 series, and evaluates Quarter-Kelly betting edges when `ODDS_API_KEY` is configured.
- **`src/tims_grouping.py`** — Pure-Python grouping/sorting for Tim Hortons candidates by group, injury status, and projected goal probability (`group_tims_candidates`), avoiding pandas `NaN` JSON coercion.
- **`src/tims_timing.py`** — Backward-induction optimal-stopping advisor (`SELECT NOW` vs `SELECTIVE DEFER`) estimating the expected value of locking picks now versus waiting for later puck-drop waves, and returning the per-group ticket to lock (`selection`) at the recommended wave. Future waves keep the contest's real group membership: each simulated pick is the best of that group's still-eligible skaters plus a few tier-matched late additions, so one ticket can never take two Group 1 stars.
- **`src/tims_tracker.py`** — Forward logger and automatic next-day NHL boxscore grader maintaining cumulative AI Hit Rate vs. Random baselines in `docs/tims_history.json`. It records the deferral-aware ticket (not the raw group leaders), freezes it once the advised lock wave passes, and exposes the lock metadata for the dashboard.
- **`src/report_builder.py` & `docs/index.html`** — Exports strictly finite JSON (`allow_nan=False`) to `docs/data.json` and styled run transcripts to `docs/latest_run.html`, powering the mobile-hardened GitHub Pages dashboard.

### Scheduling, Live Loop & Freshness Watchdog
- **`scripts/gas_hockeypicks_dispatcher.gs`** — External 24/7 Google Apps Script time-driven scheduler dispatching GitHub Actions workflows via `workflow_dispatch` on exact `America/New_York` (ET) wall-clock times.
- **`src/infer_gate.py`** — Lightweight pre-flight gate invoked by `hockeypicks_infer.yml` and `hockeypicks_live_loop.yml`. Distinguishes unconditional manual triggers (`event_name="workflow_dispatch"` with empty `cron`) from slotted external odds triggers (evaluating the `11:00 AM ET` weekend/US-CA holiday/early-matinee filter), and checks `+3 min` post-puck-drop game updates (`DEFAULT_MAX_LAG_MINUTES = 180`).
- **`src/live_loop.py`** — Long-running in-window loop executing inside GitHub Actions for matinee (`12:15–4:15 PM ET`), evening (`6:40–9:45 PM ET`), and late (`10:15–11:40 PM ET`) slates. Checks `infer_gate` every 5 minutes (`--interval-seconds 300`), commits/rebases updates when a game is due, and exits early if no non-final games fall in that window.
- **`src/health_check.py`** — Watchdog verifying JSON finiteness (`NaN`/`Infinity` rejection) and checking two staleness rules: (1) overdue `+3 min` post-start update after a `20 min` grace period, and (2) frozen `LIVE`/`CRIT` snapshots older than `45 min`.

---

## External Google Apps Script Dispatcher

### Why External `workflow_dispatch` Replaced `on: schedule:`
GitHub Actions' native `on: schedule:` runs on a best-effort shared cron queue with no latency SLA, frequently delaying jobs by hours or silently dropping ticks (e.g., missed reactive ticks and delayed daily runs). By moving scheduling out of GitHub's cron queue into a 24/7 Google Apps Script trigger calling GitHub's REST API (`POST /repos/hoskiphotoshare-boop/hockeypicks-auto/actions/workflows/<workflow>/dispatches`), runners start reliably within seconds at exact wall-clock times.

### How `hockeypicksCronTick()` Works
Deployed inside Google Apps Script project `mikeCalendars` (`hockeypicksDispatcher` file, script ID `1-N3o9sidSvLnx7c3j-wLGCwhPpocKro_X6f8XsQDo2WexOgQtvmWvm5H`):

1. **1-Minute Time-Driven Trigger**: `ScriptApp.newTrigger("hockeypicksCronTick").timeBased().everyMinutes(1).create()` executes `hockeypicksCronTick()` every minute in Google's cloud.
2. **ET Wall-Clock Conversion**: Formats `new Date()` in `"America/New_York"` via `Utilities.formatDate(...)`, automatically tracking both **EST** and **EDT** without manual offset math or dual cron rows.
3. **Zero-Overhead Fast-Path Exit**: Checks the current ET minute against all slot windows (`[slotMinute, slotMinute + 10m]`) and the hourly health window (`:17–:25`). Outside active windows, the function returns immediately in memory without touching `PropertiesService` or `UrlFetchApp`.
4. **Lock & Exactly-Once Deduplication**: Inside an active window, acquires `LockService.getScriptLock()` and checks `PropertiesService.getScriptProperties()`:
   - Daily slots store `HP_FIRED_<slotId> = "YYYY-MM-DD"` (ET date).
   - Hourly health check stores `HP_FIRED_health_hourly = "YYYY-MM-DDTHH"`.
   - Even with a 10-minute catch-up window, each slot fires **at most once** per ET day (or hour for health check).
5. **Pre-Flight Active Loop Check (`skipIfActive: true`)**: Before dispatching backup `loop_*` wake-ups, queries GitHub `GET .../actions/workflows/hockeypicks_live_loop.yml/runs?per_page=5`. If a run is already `in_progress`, `queued`, or `waiting`, GAS marks that backup slot handled for the day and skips dispatching a duplicate queued runner.

---

## Complete Dispatch Schedule

All times are evaluated in **`America/New_York`** (with corresponding Vancouver / Pacific Time shown for reference):

### Daily Slots

| Slot ID | Eastern Time (ET) | Vancouver Time (PT) | Workflow | Trigger Behavior & Purpose |
|---|---|---|---|---|
| `run_0001` | **12:01 AM ET** | **9:01 PM PT** *(prev day)* | `hockeypicks_run.yml` | **Full Retrain + Inference**: Post-midnight Tim Hortons release & prior-day grading |
| `run_1017` | **10:17 AM ET** | **7:17 AM PT** | `hockeypicks_run.yml` | **Full Retrain + Inference**: Morning skate, injury & starter updates |
| `infer_1100` | **11:00 AM ET** | **8:00 AM PT** | `hockeypicks_infer.yml` | **Inference (`cron: 0 11 * * *`)**: Weekend/holiday/early-matinee (`< 4:00 PM ET`) odds release *(skips idle weekdays)* |
| `infer_1200` | **12:00 PM ET** | **9:00 AM PT** | `hockeypicks_infer.yml` | **Inference (`cron: 0 12 * * *`)**: Daily noon odds release & mid-day line movements |
| `loop_1215` | **12:15 PM ET** | **9:15 AM PT** | `hockeypicks_live_loop.yml` | **Matinee Live Loop (`12:15–4:15 PM ET`)**: 5-minute gate checks *(exits immediately if no matinee games)* |
| `loop_1315` | **1:15 PM ET** | **10:15 AM PT** | `hockeypicks_live_loop.yml` | **Matinee Loop Backup** *(skipped if `hockeypicks_live_loop.yml` already active)* |
| `loop_1415` | **2:15 PM ET** | **11:15 AM PT** | `hockeypicks_live_loop.yml` | **Matinee Loop Backup** *(skipped if `hockeypicks_live_loop.yml` already active)* |
| `run_1617` | **4:17 PM ET** | **1:17 PM PT** | `hockeypicks_run.yml` | **Full Retrain + Inference**: Afternoon pre-game model retrain & scoring |
| `infer_1630` | **4:30 PM ET** | **1:30 PM PT** | `hockeypicks_infer.yml` | **Inference (`cron: 30 16 * * *`)**: Daily pre-game line movements & confirmed goalies |
| `loop_1840` | **6:40 PM ET** | **3:40 PM PT** | `hockeypicks_live_loop.yml` | **Evening Live Loop (`6:40–9:45 PM ET`)**: 5-minute post-puck-drop updates |
| `loop_1940` | **7:40 PM ET** | **4:40 PM PT** | `hockeypicks_live_loop.yml` | **Evening Loop Backup** *(skipped if `hockeypicks_live_loop.yml` already active)* |
| `loop_2040` | **8:40 PM ET** | **5:40 PM PT** | `hockeypicks_live_loop.yml` | **Evening Loop Backup** *(skipped if `hockeypicks_live_loop.yml` already active)* |
| `infer_2200` | **10:00 PM ET** | **7:00 PM PT** | `hockeypicks_infer.yml` | **Inference (`cron: 0 22 * * *`)**: Next-day opening odds release & late line movements |
| `loop_2215` | **10:15 PM ET** | **7:15 PM PT** | `hockeypicks_live_loop.yml` | **Late Live Loop (`10:15–11:40 PM ET`)**: Late West Coast puck-drop updates |
| `loop_2315` | **11:15 PM ET** | **8:15 PM PT** | `hockeypicks_live_loop.yml` | **Late Loop Backup** *(skipped if `hockeypicks_live_loop.yml` already active)* |

### Hourly Watchdog

| Slot ID | Schedule (ET & PT) | Workflow | Purpose |
|---|---|---|---|
| `health_hourly` | **Hourly at `:17`** *(window `:17–:25`)* | `hockeypicks_health.yml` | Verifies data freshness and opens/closes the alert GitHub Issue |

---

## Workflow Coordination & Concurrency Locks

1. **Repo-Write Lock (`concurrency.group: hockeypicks-repo-write`, `cancel-in-progress: false`)**:
   Shared by `hockeypicks_run.yml`, `hockeypicks_infer.yml`, and `hockeypicks_live_loop.yml` so concurrent runs never race when committing `docs/data.json`, `docs/tims_history.json`, `docs/latest_run.html`, or `models/`.
2. **Intentional Live-Loop Gaps**:
   `src/live_loop.py` terminates its three game windows at `4:15 PM ET`, `9:45 PM ET`, and `11:40 PM ET`, leaving clear gaps before the fixed `4:17 PM` full pipeline, `4:30 PM` pre-game odds run, `10:00 PM` opening odds run, and `12:01 AM` post-midnight run so odds releases never wait behind an idle sleep loop.
3. **Independent Health Lock (`concurrency.group: hockeypicks-health`)**:
   `hockeypicks_health.yml` only requires read access to `docs/data.json` (`contents: read`, `issues: write`), so it runs independently without blocking or queuing behind data-writing jobs.

---

## Managing the GAS Dispatcher

The `mikeCalendars` Web App deployment (`AKfycbzuBya6hA8x3_JmkTCxyFQkBe6AuKsqAkbL5AXdz-I`, restricted to `access: MYSELF`, `executeAs: USER_ACCESSING`) exposes management endpoints routed via `handleHockeypicksDoGet(e)` when called with a valid Google OAuth `Authorization: Bearer <access_token>` header:

- **Check Dispatcher Status, Trigger & Slot History**:
  ```text
  GET https://script.google.com/macros/s/AKfycbzuBya6hA8x3_JmkTCxyFQkBe6AuKsqAkbL5AXdz-I/exec?action=hockeypicks_status
  ```
  Returns `triggerInstalled`, `triggerCount`, `hasGithubToken` (boolean only; never exposes the PAT), `etTime`, `healthLastHourFired`, `lastDispatch`, and per-slot `firedToday` status.

- **Rotate `GITHUB_TOKEN` or Re-Install the 1-Minute Trigger**:
  ```text
  GET https://script.google.com/macros/s/AKfycbzuBya6hA8x3_JmkTCxyFQkBe6AuKsqAkbL5AXdz-I/exec?action=hockeypicks_setup&token=<FINE_GRAINED_PAT_ACTIONS_WRITE>
  ```
  Saves the fine-grained PAT (requires `Actions: Read and write` on `hockeypicks-auto`) in `ScriptProperties`, removes duplicate `hockeypicksCronTick` triggers, and installs a fresh 1-minute trigger.

- **Trigger an On-Demand Dispatch**:
  ```text
  GET https://script.google.com/macros/s/AKfycbzuBya6hA8x3_JmkTCxyFQkBe6AuKsqAkbL5AXdz-I/exec?action=hockeypicks_dispatch&workflow=hockeypicks_infer.yml&cron=0%2012%20*%20*%20*
  ```

- **Manually Run a Gate Tick Check**:
  ```text
  GET https://script.google.com/macros/s/AKfycbzuBya6hA8x3_JmkTCxyFQkBe6AuKsqAkbL5AXdz-I/exec?action=hockeypicks_tick
  ```

---

## GitHub Repository Setup & Local Usage

### One-Time GitHub Setup
1. Enable **GitHub Pages**: go to **Settings → Pages → Build and deployment → Source: Deploy from a branch**, select **`main`** and folder **`/docs`**, and click **Save**.
2. *(Optional)* Enable sharp betting odds & Quarter-Kelly calculations: go to **Settings → Secrets and variables → Actions → New repository secret** and add **`ODDS_API_KEY`**.
3. Ensure **`Actions` (`actions: write`)** PAT is configured in GAS via `?action=hockeypicks_setup&token=<PAT>`.

### Local Usage & Testing
```bash
pip install -r requirements.txt
python main_pipeline.py --mode full   # train + infer
python main_pipeline.py --mode train  # training only -> models/
python main_pipeline.py --mode infer  # live inference only -> docs/
python -m unittest discover -s tests -v
```

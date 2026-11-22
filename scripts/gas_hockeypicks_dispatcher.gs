/**
 * Hockeypicks AI External GitHub Actions Dispatcher (Google Apps Script)
 *
 * Replaces GitHub Actions' best-effort `on: schedule:` queue (which frequently
 * delays triggers by hours or drops them silently) with an external 24/7
 * Apps Script time-driven dispatcher that calls workflow_dispatch on exact
 * America/New_York wall-clock times (automatically handling EDT/EST).
 *
 * Designed as pure top-level Google Apps Script (V8) without CommonJS wrappers.
 *
 * Setup:
 *   Call `setupHockeypicksTriggers("<fine-grained-PAT-with-actions-write>")` once
 *   (or hit `?action=hockeypicks_setup&token=<PAT>` on the Web App endpoint).
 */

var HP_CONFIG = {
  owner: "hoskiphotoshare-boop",
  repo: "hockeypicks-auto",
  ref: "main",
  timeZone: "America/New_York",
  catchupWindowMinutes: 10,
  tokenProperty: "GITHUB_TOKEN",
  lastDispatchProperty: "HP_LAST_DISPATCH",
  firedPrefix: "HP_FIRED_",
  healthWorkflow: "hockeypicks_health.yml",
  healthStartMinute: 17,
  healthEndMinute: 25,
  liveWorkflow: "hockeypicks_live_loop.yml",
  waveCacheProperty: "HP_WAVE_CACHE",
  waveCacheTtlMs: 15 * 60 * 1000,
  waveRecheckBeforeFireMs: 2 * 60 * 1000,
  preLockLeadMs: 20 * 60 * 1000,
  postStartDelayMs: 3 * 60 * 1000,
  postStartMaxLagMs: 180 * 60 * 1000,
  liveRefreshMs: 35 * 60 * 1000,
  liveRetryCooldownMs: 20 * 60 * 1000,
  pagesDataUrl: "https://hoskiphotoshare-boop.github.io/hockeypicks-auto/data.json",
  rawDataUrl: "https://raw.githubusercontent.com/hoskiphotoshare-boop/hockeypicks-auto/main/docs/data.json"
};

var HP_ALLOWED_WORKFLOWS = {
  "hockeypicks_run.yml": true,
  "hockeypicks_infer.yml": true,
  "hockeypicks_live_loop.yml": true,
  "hockeypicks_health.yml": true,
  "hockeypicks_futures.yml": true
};

/**
 * America/New_York fixed wall-clock dispatch slots (daily unless `dayOfWeek` 1..7 Mon..Sun is set).
 * Live pre-lock, post-puck-drop (+3m), and 35m in-game refreshes are dispatched dynamically as
 * one-shot `--once` runs from docs/data.json waves rather than long-running sleep loops.
 */
var HP_SCHEDULE_SLOTS = [
  { id: "run_0001", hour: 0, minute: 1, workflow: "hockeypicks_run.yml", inputs: {} },
  { id: "run_1017", hour: 10, minute: 17, workflow: "hockeypicks_run.yml", inputs: {} },
  { id: "futures_mon_1030", dayOfWeek: 1, hour: 10, minute: 30, workflow: "hockeypicks_futures.yml", inputs: {} },
  { id: "infer_1100", hour: 11, minute: 0, workflow: "hockeypicks_infer.yml", inputs: { cron: "0 11 * * *" } },
  { id: "infer_1200", hour: 12, minute: 0, workflow: "hockeypicks_infer.yml", inputs: { cron: "0 12 * * *" } },
  { id: "run_1617", hour: 16, minute: 17, workflow: "hockeypicks_run.yml", inputs: {} },
  { id: "infer_1630", hour: 16, minute: 30, workflow: "hockeypicks_infer.yml", inputs: { cron: "30 16 * * *" } },
  { id: "infer_2200", hour: 22, minute: 0, workflow: "hockeypicks_infer.yml", inputs: { cron: "0 22 * * *" } }
];

/**
 * 1-minute time-driven entry point.
 * Fast-path returns immediately in memory outside fixed slot windows and wave triggers so
 * ScriptProperties and UrlFetchApp are only touched when a sync or dispatch is due.
 */
function hockeypicksCronTick() {
  var now = new Date();
  var nowMs = now.getTime();
  var formatted = Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd|HH|mm|HH:mm z|u");
  var parts = formatted.split("|");
  var dateStr = parts[0];
  var hourStr = parts[1];
  var hour = parseInt(hourStr, 10);
  var minute = parseInt(parts[2], 10);
  var etLabel = dateStr + " " + parts[3];
  var dayOfWeek = parseInt(parts[4], 10);
  var currentTotalMin = hour * 60 + minute;

  var matchingSlots = [];
  for (var i = 0; i < HP_SCHEDULE_SLOTS.length; i++) {
    var slot = HP_SCHEDULE_SLOTS[i];
    if (slot.dayOfWeek !== undefined && slot.dayOfWeek !== dayOfWeek) {
      continue;
    }
    var slotTotalMin = slot.hour * 60 + slot.minute;
    var diff = currentTotalMin - slotTotalMin;
    if (diff >= 0 && diff <= HP_CONFIG.catchupWindowMinutes) {
      matchingSlots.push(slot);
    }
  }

  var inHealthWindow = (minute >= HP_CONFIG.healthStartMinute && minute <= HP_CONFIG.healthEndMinute);
  var waveCache = hpReadWaveCache_();
  var needsWaveSync = (
    !waveCache ||
    waveCache.date !== dateStr ||
    typeof waveCache.fetchedAtMs !== "number" ||
    (nowMs - waveCache.fetchedAtMs) >= HP_CONFIG.waveCacheTtlMs
  );
  var waveCheck = waveCache
    ? hpFindDueWaveTriggers_(waveCache, nowMs)
    : { dueTriggers: [], liveRefreshDue: false };

  // Fast-path exit outside fixed windows, wave triggers, and 15m sync intervals
  if (
    matchingSlots.length === 0 &&
    !inHealthWindow &&
    !needsWaveSync &&
    waveCheck.dueTriggers.length === 0 &&
    !waveCheck.liveRefreshDue
  ) {
    return {
      ok: true,
      fastPathExit: true,
      etTime: etLabel,
      dispatched: []
    };
  }

  var lock = LockService.getScriptLock();
  if (!lock.tryLock(10000)) {
    return {
      ok: false,
      error: "Could not acquire script lock",
      etTime: etLabel
    };
  }

  try {
    var props = PropertiesService.getScriptProperties();
    var token = hpGetToken_(props);
    if (!token) {
      Logger.log("Hockeypicks dispatcher error: GITHUB_TOKEN is not set in ScriptProperties.");
      return {
        ok: false,
        error: "GITHUB_TOKEN is not configured in ScriptProperties",
        etTime: etLabel
      };
    }

    var actions = [];

    for (var j = 0; j < matchingSlots.length; j++) {
      var s = matchingSlots[j];
      var propKey = HP_CONFIG.firedPrefix + s.id;
      if (props.getProperty(propKey) === dateStr) {
        continue;
      }

      if (s.skipIfActive) {
        var activeCheck = hpCheckWorkflowActive_(s.workflow, token);
        if (!activeCheck.ok) {
          actions.push({
            slotId: s.id,
            workflow: s.workflow,
            status: "active_check_error",
            detail: activeCheck.error
          });
          continue;
        }
        if (activeCheck.active) {
          props.setProperty(propKey, dateStr);
          var skipRecord = {
            slotId: s.id,
            workflow: s.workflow,
            status: "skipped_already_active",
            activeRunId: activeCheck.runId,
            activeStatus: activeCheck.runStatus,
            etTime: etLabel
          };
          props.setProperty(HP_CONFIG.lastDispatchProperty, JSON.stringify(skipRecord));
          Logger.log("Skipped " + s.id + " (" + s.workflow + "): run " + activeCheck.runId + " is " + activeCheck.runStatus);
          actions.push(skipRecord);
          continue;
        }
      }

      var dispatchRes = hpDispatchWorkflow_(s.workflow, s.inputs, token);
      if (dispatchRes.ok) {
        props.setProperty(propKey, dateStr);
        var record = {
          slotId: s.id,
          workflow: s.workflow,
          inputs: s.inputs,
          status: "dispatched",
          statusCode: dispatchRes.statusCode,
          etTime: etLabel
        };
        props.setProperty(HP_CONFIG.lastDispatchProperty, JSON.stringify(record));
        Logger.log("Dispatched " + s.id + " -> " + s.workflow + " (" + dispatchRes.statusCode + ") at " + etLabel);
        actions.push(record);
      } else {
        Logger.log("Failed dispatching " + s.id + " -> " + s.workflow + ": " + dispatchRes.error);
        actions.push({
          slotId: s.id,
          workflow: s.workflow,
          status: "dispatch_error",
          statusCode: dispatchRes.statusCode,
          error: dispatchRes.error
        });
      }
    }

    if (inHealthWindow) {
      var healthKey = HP_CONFIG.firedPrefix + "health_hourly";
      var hourStamp = dateStr + "T" + hourStr;
      if (props.getProperty(healthKey) !== hourStamp) {
        var hRes = hpDispatchWorkflow_(HP_CONFIG.healthWorkflow, {}, token);
        if (hRes.ok) {
          props.setProperty(healthKey, hourStamp);
          var hRecord = {
            slotId: "health_hourly",
            workflow: HP_CONFIG.healthWorkflow,
            inputs: {},
            status: "dispatched",
            statusCode: hRes.statusCode,
            hourStamp: hourStamp,
            etTime: etLabel
          };
          props.setProperty(HP_CONFIG.lastDispatchProperty, JSON.stringify(hRecord));
          Logger.log("Dispatched health check (" + hRes.statusCode + ") for " + hourStamp);
          actions.push(hRecord);
        } else {
          Logger.log("Failed dispatching health check: " + hRes.error);
          actions.push({
            slotId: "health_hourly",
            workflow: HP_CONFIG.healthWorkflow,
            status: "dispatch_error",
            statusCode: hRes.statusCode,
            error: hRes.error
          });
        }
      }
    }

    // Re-read wave cache inside lock in case another execution just refreshed it
    waveCache = hpReadWaveCache_(props) || waveCache;
    var lockedNeedsSync = (
      !waveCache ||
      waveCache.date !== dateStr ||
      typeof waveCache.fetchedAtMs !== "number" ||
      (nowMs - waveCache.fetchedAtMs) >= HP_CONFIG.waveCacheTtlMs
    );
    var lockedCheck = waveCache
      ? hpFindDueWaveTriggers_(waveCache, nowMs)
      : { dueTriggers: [], liveRefreshDue: false };

    if (lockedNeedsSync || lockedCheck.dueTriggers.length > 0 || lockedCheck.liveRefreshDue) {
      var shouldFetchFresh = lockedNeedsSync || (
        (lockedCheck.dueTriggers.length > 0 || lockedCheck.liveRefreshDue) &&
        (!waveCache || (nowMs - waveCache.fetchedAtMs) >= HP_CONFIG.waveRecheckBeforeFireMs)
      );

      if (shouldFetchFresh) {
        var fetchRes = hpFetchDashboardJson_(token);
        if (fetchRes.ok) {
          waveCache = hpBuildWaveCache_(fetchRes.data, dateStr, nowMs, waveCache, fetchRes.source);
          hpSaveWaveCache_(waveCache, props);
          Logger.log(
            "Synced wave schedule (" + fetchRes.source + "): " +
            waveCache.triggers.length + " triggers, hasLiveGames=" + waveCache.hasLiveGames
          );
        } else {
          Logger.log("Wave schedule sync warning: " + fetchRes.error);
          actions.push({
            slotId: "wave_sync",
            status: "sync_warning",
            error: fetchRes.error
          });
        }
      }

      var freshCheck = waveCache
        ? hpFindDueWaveTriggers_(waveCache, nowMs)
        : { dueTriggers: [], liveRefreshDue: false };

      if (freshCheck.dueTriggers.length > 0 || freshCheck.liveRefreshDue) {
        var triggerLabels = [];
        for (var k = 0; k < freshCheck.dueTriggers.length; k++) {
          triggerLabels.push(freshCheck.dueTriggers[k].label);
        }
        if (freshCheck.liveRefreshDue) {
          triggerLabels.push("35m live refresh (" + (waveCache.liveLabels || []).join(", ") + ")");
        }

        var waveActive = hpCheckWorkflowActive_(HP_CONFIG.liveWorkflow, token);
        if (!waveActive.ok) {
          actions.push({
            slotId: "dynamic_wave",
            workflow: HP_CONFIG.liveWorkflow,
            status: "active_check_error",
            detail: waveActive.error
          });
        } else if (waveActive.active) {
          var waveSkip = {
            slotId: "dynamic_wave",
            workflow: HP_CONFIG.liveWorkflow,
            status: "skipped_already_active",
            activeRunId: waveActive.runId,
            activeStatus: waveActive.runStatus,
            triggers: triggerLabels,
            etTime: etLabel
          };
          Logger.log("Skipped dynamic_wave (" + HP_CONFIG.liveWorkflow + "): run " + waveActive.runId + " is " + waveActive.runStatus);
          actions.push(waveSkip);
        } else {
          var waveInputs = { once: true, dry_run: false };
          var wRes = hpDispatchWorkflow_(HP_CONFIG.liveWorkflow, waveInputs, token);
          if (wRes.ok) {
            if (!waveCache.firedIds) {
              waveCache.firedIds = {};
            }
            for (var fIdx = 0; fIdx < freshCheck.dueTriggers.length; fIdx++) {
              waveCache.firedIds[freshCheck.dueTriggers[fIdx].id] = nowMs;
            }
            if (freshCheck.liveRefreshDue) {
              waveCache.lastLiveDispatchMs = nowMs;
            }
            hpSaveWaveCache_(waveCache, props);

            var wRecord = {
              slotId: "dynamic_wave",
              workflow: HP_CONFIG.liveWorkflow,
              inputs: waveInputs,
              triggers: triggerLabels,
              status: "dispatched",
              statusCode: wRes.statusCode,
              etTime: etLabel
            };
            props.setProperty(HP_CONFIG.lastDispatchProperty, JSON.stringify(wRecord));
            Logger.log("Dispatched dynamic_wave --once -> " + HP_CONFIG.liveWorkflow + " (" + triggerLabels.join("; ") + ")");
            actions.push(wRecord);
          } else {
            Logger.log("Failed dispatching dynamic_wave: " + wRes.error);
            actions.push({
              slotId: "dynamic_wave",
              workflow: HP_CONFIG.liveWorkflow,
              status: "dispatch_error",
              statusCode: wRes.statusCode,
              error: wRes.error
            });
          }
        }
      }
    }

    return {
      ok: true,
      fastPathExit: false,
      etTime: etLabel,
      actions: actions
    };
  } finally {
    lock.releaseLock();
  }
}

/**
 * Installs (or replaces) the 1-minute time-driven trigger for hockeypicksCronTick
 * and optionally saves GITHUB_TOKEN in ScriptProperties.
 */
function setupHockeypicksTriggers(githubToken) {
  var props = PropertiesService.getScriptProperties();
  if (githubToken && String(githubToken).trim()) {
    props.setProperty(HP_CONFIG.tokenProperty, String(githubToken).trim());
  }

  var existing = ScriptApp.getProjectTriggers();
  var removed = 0;
  for (var i = 0; i < existing.length; i++) {
    if (existing[i].getHandlerFunction() === "hockeypicksCronTick") {
      ScriptApp.deleteTrigger(existing[i]);
      removed++;
    }
  }

  var created = ScriptApp.newTrigger("hockeypicksCronTick")
    .timeBased()
    .everyMinutes(1)
    .create();

  return {
    ok: true,
    removedOldTriggers: removed,
    triggerInstalled: true,
    triggerId: created.getUniqueId(),
    handlerFunction: "hockeypicksCronTick",
    intervalMinutes: 1,
    hasGithubToken: Boolean(hpGetToken_(props)),
    etTime: Utilities.formatDate(new Date(), HP_CONFIG.timeZone, "yyyy-MM-dd HH:mm:ss z")
  };
}

/**
 * Returns the dispatcher configuration, installed trigger state, token readiness
 * (boolean only; never exposes the token), fixed slot execution status, and
 * cached dynamic game-wave triggers.
 */
function getHockeypicksDispatcherStatus() {
  var now = new Date();
  var nowMs = now.getTime();
  var todayEt = Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd");
  var etTime = Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd HH:mm:ss z");

  var triggers = ScriptApp.getProjectTriggers();
  var matchingTriggerIds = [];
  for (var i = 0; i < triggers.length; i++) {
    if (triggers[i].getHandlerFunction() === "hockeypicksCronTick") {
      matchingTriggerIds.push(triggers[i].getUniqueId());
    }
  }

  var rawProps = PropertiesService.getScriptProperties();
  var props = rawProps.getProperties();
  var hasToken = Boolean(props[HP_CONFIG.tokenProperty] || props.HP_GITHUB_TOKEN);

  var slots = [];
  for (var j = 0; j < HP_SCHEDULE_SLOTS.length; j++) {
    var s = HP_SCHEDULE_SLOTS[j];
    var key = HP_CONFIG.firedPrefix + s.id;
    slots.push({
      id: s.id,
      dayOfWeek: s.dayOfWeek !== undefined ? s.dayOfWeek : null,
      timeEt: (s.hour < 10 ? "0" : "") + s.hour + ":" + (s.minute < 10 ? "0" : "") + s.minute,
      workflow: s.workflow,
      skipIfActive: Boolean(s.skipIfActive),
      lastFiredDate: props[key] || null,
      firedToday: props[key] === todayEt
    });
  }

  var lastDispatch = null;
  if (props[HP_CONFIG.lastDispatchProperty]) {
    try {
      lastDispatch = JSON.parse(props[HP_CONFIG.lastDispatchProperty]);
    } catch (e) {
      lastDispatch = props[HP_CONFIG.lastDispatchProperty];
    }
  }

  var waveCache = hpReadWaveCache_(rawProps);
  var waveSchedule = hpFormatWaveStatus_(waveCache, nowMs);

  return {
    ok: true,
    repo: HP_CONFIG.owner + "/" + HP_CONFIG.repo,
    ref: HP_CONFIG.ref,
    timeZone: HP_CONFIG.timeZone,
    etTime: etTime,
    triggerInstalled: matchingTriggerIds.length > 0,
    triggerCount: matchingTriggerIds.length,
    triggerIds: matchingTriggerIds,
    hasGithubToken: hasToken,
    healthLastHourFired: props[HP_CONFIG.firedPrefix + "health_hourly"] || null,
    lastDispatch: lastDispatch,
    slots: slots,
    waveSchedule: waveSchedule
  };
}

/**
 * Web App request router for hockeypicks_* actions inside doGet(e).
 */
function handleHockeypicksDoGet(e) {
  var params = (e && e.parameter) || e || {};
  var action = params.action || "hockeypicks_status";
  var result;

  if (action === "hockeypicks_status") {
    result = getHockeypicksDispatcherStatus();
  } else if (action === "hockeypicks_setup") {
    if (!params.__via_post) {
      result = { ok: false, error: "hockeypicks_setup requires POST with a JSON body (never send the token in the URL)." };
    } else if (!String(params.token || "").trim()) {
      result = { ok: false, error: "A non-empty token is required to rotate the GitHub token." };
    } else {
      result = {
        setup: setupHockeypicksTriggers(params.token || ""),
        status: getHockeypicksDispatcherStatus()
      };
    }
  } else if (action === "hockeypicks_tick") {
    result = hockeypicksCronTick();
  } else if (action === "hockeypicks_sync_waves") {
    var lock = LockService.getScriptLock();
    if (!lock.tryLock(10000)) {
      result = { ok: false, error: "Could not acquire script lock" };
    } else {
      try {
        var now = new Date();
        var nowMs = now.getTime();
        var dateStr = Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd");
        var props = PropertiesService.getScriptProperties();
        var token = hpGetToken_(props);
        var prevCache = hpReadWaveCache_(props);
        var fetchRes = hpFetchDashboardJson_(token);
        if (!fetchRes.ok) {
          result = { ok: false, error: fetchRes.error };
        } else {
          var nextCache = hpBuildWaveCache_(fetchRes.data, dateStr, nowMs, prevCache, fetchRes.source);
          hpSaveWaveCache_(nextCache, props);
          result = {
            ok: true,
            etTime: Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd HH:mm:ss z"),
            sync: {
              source: fetchRes.source,
              fetchedAtEt: hpFormatEtMs_(nextCache.fetchedAtMs),
              triggerCount: nextCache.triggers.length,
              hasLiveGames: nextCache.hasLiveGames
            },
            status: getHockeypicksDispatcherStatus()
          };
        }
      } finally {
        lock.releaseLock();
      }
    }
  } else if (action === "hockeypicks_dispatch") {
    var workflow = String(params.workflow || "").trim();
    if (!HP_ALLOWED_WORKFLOWS[workflow]) {
      result = {
        ok: false,
        error: "Unsupported workflow parameter",
        allowed: Object.keys(HP_ALLOWED_WORKFLOWS)
      };
    } else {
      var dispatchProps = PropertiesService.getScriptProperties();
      var dispatchToken = hpGetToken_(dispatchProps);
      if (!dispatchToken) {
        result = { ok: false, error: "GITHUB_TOKEN is not configured in ScriptProperties" };
      } else {
        var inputs = {};
        if (workflow === "hockeypicks_infer.yml" && params.cron) {
          inputs.cron = String(params.cron);
        } else if (workflow === "hockeypicks_live_loop.yml") {
          if (params.once !== undefined) inputs.once = (params.once === "true");
          if (params.dry_run !== undefined) inputs.dry_run = (params.dry_run === "true");
        } else if (workflow === "hockeypicks_futures.yml") {
          if (params.bankroll !== undefined) inputs.bankroll = String(params.bankroll);
          if (params.n_sims !== undefined) inputs.n_sims = String(params.n_sims);
        }
        var dispatchRes = hpDispatchWorkflow_(workflow, inputs, dispatchToken);
        var etLabel = Utilities.formatDate(new Date(), HP_CONFIG.timeZone, "yyyy-MM-dd HH:mm:ss z");
        if (dispatchRes.ok) {
          dispatchProps.setProperty(HP_CONFIG.lastDispatchProperty, JSON.stringify({
            slotId: "manual_web_dispatch",
            workflow: workflow,
            inputs: inputs,
            status: "dispatched",
            statusCode: dispatchRes.statusCode,
            etTime: etLabel
          }));
        }
        result = {
          ok: dispatchRes.ok,
          workflow: workflow,
          inputs: inputs,
          statusCode: dispatchRes.statusCode,
          error: dispatchRes.error || null,
          etTime: etLabel
        };
      }
    }
  } else {
    result = { ok: false, error: "Unknown hockeypicks action: " + action };
  }

  return ContentService.createTextOutput(JSON.stringify(result, null, 2))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Retrieves the GitHub PAT from ScriptProperties without ever returning it in logs/HTTP output.
 */
function hpGetToken_(props) {
  var p = props || PropertiesService.getScriptProperties();
  return p.getProperty(HP_CONFIG.tokenProperty) || p.getProperty("HP_GITHUB_TOKEN") || "";
}

/**
 * Parses `"YYYY-MM-DD HH:MM:SS UTC"`, `"YYYY-MM-DD HH:MM UTC"`, or ISO-8601 strings to UTC epoch ms.
 */
function hpParseUtcMs_(value) {
  if (!value || typeof value !== "string") {
    return null;
  }
  var s = value.trim();
  if (!s) {
    return null;
  }
  var m = s.match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\s*UTC|Z)?$/i);
  if (m) {
    return Date.UTC(
      parseInt(m[1], 10),
      parseInt(m[2], 10) - 1,
      parseInt(m[3], 10),
      parseInt(m[4], 10),
      parseInt(m[5], 10),
      m[6] ? parseInt(m[6], 10) : 0
    );
  }
  var ms = Date.parse(s);
  return isNaN(ms) ? null : ms;
}

/**
 * Formats an epoch millisecond timestamp in America/New_York for status diagnostics.
 */
function hpFormatEtMs_(ms) {
  if (typeof ms !== "number" || !isFinite(ms)) {
    return null;
  }
  return Utilities.formatDate(new Date(ms), HP_CONFIG.timeZone, "yyyy-MM-dd HH:mm z");
}

/**
 * Fetches docs/data.json from GitHub Contents API (0s CDN lag), falling back to GitHub Pages
 * and raw.githubusercontent.com.
 */
function hpFetchDashboardJson_(token) {
  var attempts = [];
  var apiHeaders = {
    "Accept": "application/vnd.github.raw+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "hockeypicks-gas-dispatcher"
  };
  if (token) {
    apiHeaders.Authorization = "Bearer " + token;
  }
  attempts.push({
    source: "github_api",
    url: "https://api.github.com/repos/" + HP_CONFIG.owner + "/" + HP_CONFIG.repo + "/contents/docs/data.json?ref=" + encodeURIComponent(HP_CONFIG.ref),
    headers: apiHeaders
  });
  var bust = "?t=" + Date.now();
  attempts.push({
    source: "github_pages",
    url: HP_CONFIG.pagesDataUrl + bust,
    headers: { "User-Agent": "hockeypicks-gas-dispatcher" }
  });
  attempts.push({
    source: "github_raw",
    url: HP_CONFIG.rawDataUrl + bust,
    headers: { "User-Agent": "hockeypicks-gas-dispatcher" }
  });

  var errors = [];
  for (var i = 0; i < attempts.length; i++) {
    var a = attempts[i];
    try {
      var resp = UrlFetchApp.fetch(a.url, {
        method: "get",
        muteHttpExceptions: true,
        headers: a.headers
      });
      var code = resp.getResponseCode();
      if (code >= 200 && code < 300) {
        var parsed = JSON.parse(resp.getContentText());
        if (parsed && typeof parsed === "object" && parsed.meta) {
          return { ok: true, data: parsed, source: a.source };
        }
        errors.push(a.source + ": missing meta object");
      } else {
        errors.push(a.source + ": HTTP " + code);
      }
    } catch (err) {
      errors.push(a.source + ": " + String(err));
    }
  }
  return { ok: false, error: errors.join(" | ") };
}

/**
 * Builds a compact (<2 KB) wave schedule cache from docs/data.json.
 */
function hpBuildWaveCache_(payload, dateStr, nowMs, prevCache, source) {
  var meta = (payload && payload.meta) || {};
  var dataUpdatedMs = hpParseUtcMs_(meta.updated_utc);

  var firedIds = {};
  if (prevCache && prevCache.firedIds) {
    var cutoffMs = nowMs - 24 * 60 * 60 * 1000;
    var prevKeys = Object.keys(prevCache.firedIds);
    for (var i = 0; i < prevKeys.length; i++) {
      var k = prevKeys[i];
      var ts = prevCache.firedIds[k];
      if (typeof ts === "number" && ts >= cutoffMs) {
        firedIds[k] = ts;
      }
    }
  }

  var lastLiveDispatchMs = null;
  if (prevCache && typeof prevCache.lastLiveDispatchMs === "number" && (nowMs - prevCache.lastLiveDispatchMs) < 24 * 60 * 60 * 1000) {
    lastLiveDispatchMs = prevCache.lastLiveDispatchMs;
  }

  var triggers = [];
  var timing = (payload && payload.tims_timing) || {};
  var lockUtc = timing.selection_lock_utc;
  var lockMs = hpParseUtcMs_(lockUtc);
  if (lockMs !== null) {
    triggers.push({
      id: "prelock_" + lockUtc,
      kind: "pre_lock",
      dueMs: lockMs - HP_CONFIG.preLockLeadMs,
      expireMs: lockMs,
      label: "Pre-lock " + (timing.selection_lock_et || lockUtc)
    });
  }

  var games = (payload && payload.games) || [];
  var waveGroups = {};
  var hasLiveGames = false;
  var liveLabels = [];

  for (var gIdx = 0; gIdx < games.length; gIdx++) {
    var g = games[gIdx] || {};
    var state = String(g.game_state || "").toUpperCase();
    var startUtc = g.start_time_utc;
    var startMs = hpParseUtcMs_(startUtc);
    var matchup = (g.away || "?") + "@" + (g.home || "?");

    if ((state === "LIVE" || state === "CRIT") && (startMs === null || startMs <= nowMs)) {
      hasLiveGames = true;
      if (liveLabels.length < 3) {
        liveLabels.push(matchup);
      }
    }

    if (state === "FINAL" || state === "OFF" || startMs === null) {
      continue;
    }

    if (!waveGroups[startUtc]) {
      waveGroups[startUtc] = {
        startUtc: startUtc,
        startMs: startMs,
        et: g.start_time_et || startUtc,
        matchups: []
      };
    }
    waveGroups[startUtc].matchups.push(matchup);
  }

  var startKeys = Object.keys(waveGroups);
  for (var wIdx = 0; wIdx < startKeys.length; wIdx++) {
    var grp = waveGroups[startKeys[wIdx]];
    var shown = grp.matchups.slice(0, 3).join(", ");
    var extra = grp.matchups.length > 3 ? " +" + (grp.matchups.length - 3) : "";
    var dueMs = grp.startMs + HP_CONFIG.postStartDelayMs;
    triggers.push({
      id: "wave_" + grp.startUtc,
      kind: "post_start",
      dueMs: dueMs,
      expireMs: dueMs + HP_CONFIG.postStartMaxLagMs,
      label: grp.et + " (" + shown + extra + ")"
    });
  }

  triggers.sort(function(a, b) { return a.dueMs - b.dueMs; });

  var liveRefreshDueMs = (hasLiveGames && dataUpdatedMs !== null)
    ? (dataUpdatedMs + HP_CONFIG.liveRefreshMs)
    : null;

  return {
    date: dateStr,
    source: source || "unknown",
    fetchedAtMs: nowMs,
    dataUpdatedMs: dataUpdatedMs,
    triggers: triggers,
    hasLiveGames: hasLiveGames,
    liveLabels: liveLabels,
    liveRefreshDueMs: liveRefreshDueMs,
    lastLiveDispatchMs: lastLiveDispatchMs,
    firedIds: firedIds
  };
}

/**
 * Reads the compact wave cache from ScriptCache first, falling back to ScriptProperties.
 */
function hpReadWaveCache_(props) {
  var key = HP_CONFIG.waveCacheProperty;
  var scriptCache = CacheService.getScriptCache();
  var raw = scriptCache.get(key);
  if (!raw) {
    var p = props || PropertiesService.getScriptProperties();
    raw = p.getProperty(key);
    if (raw) {
      scriptCache.put(key, raw, 1200);
    }
  }
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

/**
 * Writes the compact wave cache to both ScriptCache (fast 1m read) and ScriptProperties (durable).
 */
function hpSaveWaveCache_(cache, props) {
  var raw = JSON.stringify(cache);
  CacheService.getScriptCache().put(HP_CONFIG.waveCacheProperty, raw, 1200);
  (props || PropertiesService.getScriptProperties()).setProperty(HP_CONFIG.waveCacheProperty, raw);
}

/**
 * Evaluates which wave triggers or 35m live refreshes are currently due and unfired.
 */
function hpFindDueWaveTriggers_(cache, nowMs) {
  var dueTriggers = [];
  if (!cache) {
    return { dueTriggers: dueTriggers, liveRefreshDue: false };
  }
  var triggers = cache.triggers || [];
  var firedIds = cache.firedIds || {};
  var dataUpdatedMs = cache.dataUpdatedMs;

  for (var i = 0; i < triggers.length; i++) {
    var t = triggers[i];
    if (firedIds[t.id]) {
      continue;
    }
    if (nowMs < t.dueMs || nowMs >= t.expireMs) {
      continue;
    }
    if (typeof dataUpdatedMs === "number" && dataUpdatedMs >= t.dueMs) {
      continue;
    }
    dueTriggers.push(t);
  }

  var liveRefreshDue = Boolean(
    cache.hasLiveGames &&
    typeof cache.liveRefreshDueMs === "number" &&
    nowMs >= cache.liveRefreshDueMs &&
    (!cache.lastLiveDispatchMs || (nowMs - cache.lastLiveDispatchMs) >= HP_CONFIG.liveRetryCooldownMs)
  );

  return {
    dueTriggers: dueTriggers,
    liveRefreshDue: liveRefreshDue
  };
}

/**
 * Formats cached wave schedule state for `hockeypicks_status`.
 */
function hpFormatWaveStatus_(cache, nowMs) {
  if (!cache) {
    return null;
  }
  var firedIds = cache.firedIds || {};
  var dataUpdatedMs = cache.dataUpdatedMs;
  var mapped = [];
  var triggers = cache.triggers || [];
  var firedCount = 0;

  for (var i = 0; i < triggers.length; i++) {
    var t = triggers[i];
    var isFired = Boolean(firedIds[t.id]);
    var covered = (typeof dataUpdatedMs === "number" && dataUpdatedMs >= t.dueMs);
    if (isFired) {
      firedCount++;
    }
    mapped.push({
      id: t.id,
      kind: t.kind,
      label: t.label,
      dueEt: hpFormatEtMs_(t.dueMs),
      expireEt: hpFormatEtMs_(t.expireMs),
      fired: isFired,
      coveredByData: covered,
      expired: nowMs >= t.expireMs
    });
  }

  return {
    date: cache.date,
    source: cache.source || null,
    fetchedAtEt: hpFormatEtMs_(cache.fetchedAtMs),
    dataUpdatedEt: hpFormatEtMs_(cache.dataUpdatedMs),
    hasLiveGames: Boolean(cache.hasLiveGames),
    liveLabels: cache.liveLabels || [],
    liveRefreshDueEt: hpFormatEtMs_(cache.liveRefreshDueMs),
    lastLiveDispatchEt: hpFormatEtMs_(cache.lastLiveDispatchMs),
    firedCount: firedCount,
    triggers: mapped
  };
}

/**
 * Checks whether a workflow currently has an `in_progress` or `queued` run.
 */
function hpCheckWorkflowActive_(workflowFile, token) {
  var url = "https://api.github.com/repos/" +
    HP_CONFIG.owner + "/" + HP_CONFIG.repo +
    "/actions/workflows/" + encodeURIComponent(workflowFile) +
    "/runs?per_page=5";

  var resp = UrlFetchApp.fetch(url, {
    method: "get",
    muteHttpExceptions: true,
    headers: {
      "Authorization": "Bearer " + token,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "hockeypicks-gas-dispatcher"
    }
  });

  var code = resp.getResponseCode();
  if (code < 200 || code >= 300) {
    return {
      ok: false,
      error: "GitHub runs query HTTP " + code + ": " + resp.getContentText().slice(0, 300)
    };
  }

  var parsed = JSON.parse(resp.getContentText());
  var runs = parsed.workflow_runs || [];
  for (var i = 0; i < runs.length; i++) {
    var st = runs[i].status;
    if (st === "in_progress" || st === "queued" || st === "waiting") {
      return {
        ok: true,
        active: true,
        runId: runs[i].id,
        runStatus: st
      };
    }
  }
  return { ok: true, active: false };
}

/**
 * Triggers a GitHub Actions workflow_dispatch event for `workflowFile` on `HP_CONFIG.ref`.
 */
function hpDispatchWorkflow_(workflowFile, inputs, token) {
  var url = "https://api.github.com/repos/" +
    HP_CONFIG.owner + "/" + HP_CONFIG.repo +
    "/actions/workflows/" + encodeURIComponent(workflowFile) +
    "/dispatches";

  var payload = { ref: HP_CONFIG.ref };
  if (inputs && Object.keys(inputs).length > 0) {
    payload.inputs = inputs;
  }

  var resp = UrlFetchApp.fetch(url, {
    method: "post",
    contentType: "application/json",
    muteHttpExceptions: true,
    headers: {
      "Authorization": "Bearer " + token,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "hockeypicks-gas-dispatcher"
    },
    payload: JSON.stringify(payload)
  });

  var code = resp.getResponseCode();
  if (code === 204 || (code >= 200 && code < 300)) {
    return { ok: true, statusCode: code };
  }
  return {
    ok: false,
    statusCode: code,
    error: "HTTP " + code + ": " + resp.getContentText().slice(0, 400)
  };
}


/**
 * Web App POST router: accepts a JSON body so the GitHub token never travels in the URL.
 */
function doPost(e) {
  var params = {};
  try {
    var body = (e && e.postData && e.postData.contents) || "{}";
    params = JSON.parse(body) || {};
  } catch (err) {
    params = {};
  }
  params.__via_post = true;
  return handleHockeypicksDoGet(params);
}

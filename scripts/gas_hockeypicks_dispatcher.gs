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
  healthEndMinute: 25
};

var HP_ALLOWED_WORKFLOWS = {
  "hockeypicks_run.yml": true,
  "hockeypicks_infer.yml": true,
  "hockeypicks_live_loop.yml": true,
  "hockeypicks_health.yml": true
};

/**
 * Daily America/New_York wall-clock dispatch slots.
 * Each slot has a 10-minute catch-up window (`catchupWindowMinutes`) and is
 * deduplicated per ET calendar date via ScriptProperties (`HP_FIRED_<id>`).
 */
var HP_SCHEDULE_SLOTS = [
  { id: "run_0001", hour: 0, minute: 1, workflow: "hockeypicks_run.yml", inputs: {} },
  { id: "run_1017", hour: 10, minute: 17, workflow: "hockeypicks_run.yml", inputs: {} },
  { id: "infer_1100", hour: 11, minute: 0, workflow: "hockeypicks_infer.yml", inputs: { cron: "0 11 * * *" } },
  { id: "infer_1200", hour: 12, minute: 0, workflow: "hockeypicks_infer.yml", inputs: { cron: "0 12 * * *" } },
  { id: "loop_1215", hour: 12, minute: 15, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true },
  { id: "loop_1315", hour: 13, minute: 15, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true },
  { id: "loop_1415", hour: 14, minute: 15, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true },
  { id: "run_1617", hour: 16, minute: 17, workflow: "hockeypicks_run.yml", inputs: {} },
  { id: "infer_1630", hour: 16, minute: 30, workflow: "hockeypicks_infer.yml", inputs: { cron: "30 16 * * *" } },
  { id: "loop_1840", hour: 18, minute: 40, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true },
  { id: "loop_1940", hour: 19, minute: 40, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true },
  { id: "loop_2040", hour: 20, minute: 40, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true },
  { id: "infer_2200", hour: 22, minute: 0, workflow: "hockeypicks_infer.yml", inputs: { cron: "0 22 * * *" } },
  { id: "loop_2215", hour: 22, minute: 15, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true },
  { id: "loop_2315", hour: 23, minute: 15, workflow: "hockeypicks_live_loop.yml", inputs: { once: false, dry_run: false }, skipIfActive: true }
];

/**
 * 1-minute time-driven entry point.
 * Fast-path returns immediately in memory outside scheduled slot windows so
 * ScriptProperties and UrlFetchApp are only touched when a dispatch is due.
 */
function hockeypicksCronTick() {
  var now = new Date();
  var formatted = Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd|HH|mm|HH:mm z");
  var parts = formatted.split("|");
  var dateStr = parts[0];
  var hourStr = parts[1];
  var hour = parseInt(hourStr, 10);
  var minute = parseInt(parts[2], 10);
  var etLabel = dateStr + " " + parts[3];
  var currentTotalMin = hour * 60 + minute;

  var matchingSlots = [];
  for (var i = 0; i < HP_SCHEDULE_SLOTS.length; i++) {
    var slot = HP_SCHEDULE_SLOTS[i];
    var slotTotalMin = slot.hour * 60 + slot.minute;
    var diff = currentTotalMin - slotTotalMin;
    if (diff >= 0 && diff <= HP_CONFIG.catchupWindowMinutes) {
      matchingSlots.push(slot);
    }
  }

  var inHealthWindow = (minute >= HP_CONFIG.healthStartMinute && minute <= HP_CONFIG.healthEndMinute);

  // Fast-path exit outside all windows to minimize daily quota consumption
  if (matchingSlots.length === 0 && !inHealthWindow) {
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
 * (boolean only; never exposes the token), and per-slot execution status.
 */
function getHockeypicksDispatcherStatus() {
  var now = new Date();
  var todayEt = Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd");
  var etTime = Utilities.formatDate(now, HP_CONFIG.timeZone, "yyyy-MM-dd HH:mm:ss z");

  var triggers = ScriptApp.getProjectTriggers();
  var matchingTriggerIds = [];
  for (var i = 0; i < triggers.length; i++) {
    if (triggers[i].getHandlerFunction() === "hockeypicksCronTick") {
      matchingTriggerIds.push(triggers[i].getUniqueId());
    }
  }

  var props = PropertiesService.getScriptProperties().getProperties();
  var hasToken = Boolean(props[HP_CONFIG.tokenProperty] || props.HP_GITHUB_TOKEN);

  var slots = [];
  for (var j = 0; j < HP_SCHEDULE_SLOTS.length; j++) {
    var s = HP_SCHEDULE_SLOTS[j];
    var key = HP_CONFIG.firedPrefix + s.id;
    slots.push({
      id: s.id,
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
    slots: slots
  };
}

/**
 * Web App request router for hockeypicks_* actions inside doGet(e).
 */
function handleHockeypicksDoGet(e) {
  var params = (e && e.parameter) || {};
  var action = params.action || "hockeypicks_status";
  var result;

  if (action === "hockeypicks_status") {
    result = getHockeypicksDispatcherStatus();
  } else if (action === "hockeypicks_setup") {
    result = {
      setup: setupHockeypicksTriggers(params.token || ""),
      status: getHockeypicksDispatcherStatus()
    };
  } else if (action === "hockeypicks_tick") {
    result = hockeypicksCronTick();
  } else if (action === "hockeypicks_dispatch") {
    var workflow = String(params.workflow || "").trim();
    if (!HP_ALLOWED_WORKFLOWS[workflow]) {
      result = {
        ok: false,
        error: "Unsupported workflow parameter",
        allowed: Object.keys(HP_ALLOWED_WORKFLOWS)
      };
    } else {
      var props = PropertiesService.getScriptProperties();
      var token = hpGetToken_(props);
      if (!token) {
        result = { ok: false, error: "GITHUB_TOKEN is not configured in ScriptProperties" };
      } else {
        var inputs = {};
        if (workflow === "hockeypicks_infer.yml" && params.cron) {
          inputs.cron = String(params.cron);
        } else if (workflow === "hockeypicks_live_loop.yml") {
          if (params.once !== undefined) inputs.once = (params.once === "true");
          if (params.dry_run !== undefined) inputs.dry_run = (params.dry_run === "true");
        }
        var dispatchRes = hpDispatchWorkflow_(workflow, inputs, token);
        var etLabel = Utilities.formatDate(new Date(), HP_CONFIG.timeZone, "yyyy-MM-dd HH:mm:ss z");
        if (dispatchRes.ok) {
          props.setProperty(HP_CONFIG.lastDispatchProperty, JSON.stringify({
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

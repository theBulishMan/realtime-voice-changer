const controlState = {
  currentSettings: {
    vad_silence_ms: 140,
    max_segment_ms: 2800,
    ptt_enabled: true,
    llm_correction_enabled: true,
    input_gain_db: 0,
    output_gain_db: 0,
    denoise_enabled: false,
    denoise_strength: 0.35,
    ambience_enabled: true,
    ambience_level: 0.003,
  },
  audioLevels: {
    input: 0,
    output: 0,
  },
  realtimeRunning: false,
  pttActive: false,
  pttPending: false,
  levelDecayTimer: null,
  settingsAutosaveTimer: null,
  settingsAutosaveInFlight: false,
};

const AUDIO_LEVEL_DECAY_STEP = 0.04;
const AUDIO_LEVEL_DECAY_MS = 120;
const MODEL_STRIP_POLL_MS = 3500;
let modelStripPollTimer = null;

function toNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function clampLevel(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  return Math.min(1, Math.max(0, num));
}

function applyLevel(kind, value) {
  const fillId = kind === "input" ? "inputLevelFill" : "outputLevelFill";
  const textId = kind === "input" ? "inputLevelText" : "outputLevelText";
  const fillEl = RVC.byId(fillId);
  const textEl = RVC.byId(textId);
  const normalized = clampLevel(value);
  const displayed = Math.pow(normalized, 0.55);
  const percent = Math.round(displayed * 100);

  if (fillEl) {
    fillEl.style.width = `${percent}%`;
    if (fillEl.parentElement) {
      fillEl.parentElement.setAttribute("aria-valuenow", String(percent));
    }
  }
  if (textEl) {
    textEl.textContent = `${percent}%`;
  }
}

function renderAudioLevels() {
  applyLevel("input", controlState.audioLevels.input);
  applyLevel("output", controlState.audioLevels.output);
}

function updateAudioLevels(data) {
  if (typeof data.input === "number") {
    controlState.audioLevels.input = clampLevel(data.input);
  }
  if (typeof data.output === "number") {
    controlState.audioLevels.output = clampLevel(data.output);
  }
  renderAudioLevels();
}

function resetAudioLevels() {
  controlState.audioLevels.input = 0;
  controlState.audioLevels.output = 0;
  renderAudioLevels();
}

function startAudioLevelDecay() {
  if (controlState.levelDecayTimer != null) {
    return;
  }
  controlState.levelDecayTimer = window.setInterval(() => {
    controlState.audioLevels.input = Math.max(0, controlState.audioLevels.input - AUDIO_LEVEL_DECAY_STEP);
    controlState.audioLevels.output = Math.max(0, controlState.audioLevels.output - AUDIO_LEVEL_DECAY_STEP);
    renderAudioLevels();
  }, AUDIO_LEVEL_DECAY_MS);
}

function stopAudioLevelDecay() {
  if (controlState.levelDecayTimer == null) {
    return;
  }
  window.clearInterval(controlState.levelDecayTimer);
  controlState.levelDecayTimer = null;
}

function renderTuningLabels() {
  const inDb = toNumber(RVC.byId("inputGainDb")?.value, 0);
  const outDb = toNumber(RVC.byId("outputGainDb")?.value, 0);
  const denoise = toNumber(RVC.byId("denoiseStrength")?.value, 0.35);
  const ambience = toNumber(RVC.byId("ambienceLevel")?.value, 0.003);

  const inText = RVC.byId("inputGainDbText");
  const outText = RVC.byId("outputGainDbText");
  const denoiseText = RVC.byId("denoiseStrengthText");
  const ambienceText = RVC.byId("ambienceLevelText");
  if (inText) inText.textContent = `${inDb.toFixed(1)} dB`;
  if (outText) outText.textContent = `${outDb.toFixed(1)} dB`;
  if (denoiseText) denoiseText.textContent = `${Math.round(Math.min(1, Math.max(0, denoise)) * 100)}%`;
  if (ambienceText) ambienceText.textContent = `${(Math.min(0.06, Math.max(0, ambience)) * 100).toFixed(2)}%`;
}

function syncPttUiState() {
  const holdBtn = RVC.byId("btnPttHold");
  const label = RVC.byId("pttStateLabel");
  const enabled = Boolean(controlState.currentSettings.ptt_enabled);
  const running = Boolean(controlState.realtimeRunning);
  const active = Boolean(controlState.pttActive);

  if (holdBtn) {
    holdBtn.disabled = !enabled || !running || controlState.pttPending;
    holdBtn.classList.toggle("active", enabled && running && active);
  }
  if (label) {
    if (!enabled) {
      label.textContent = "关闭";
    } else if (!running) {
      label.textContent = "未启动";
    } else if (active) {
      label.textContent = "讲话中";
  } else {
      label.textContent = "待机";
    }
  }
}

function isEditableElement(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tag = String(target.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}

async function setPttActive(active, options = {}) {
  const desired = Boolean(active);
  if (!controlState.realtimeRunning) {
    controlState.pttActive = false;
    syncPttUiState();
    return;
  }
  if (!controlState.currentSettings.ptt_enabled) {
    controlState.pttActive = true;
    syncPttUiState();
    return;
  }
  if (controlState.pttPending && !options.force) {
    return;
  }
  if (controlState.pttActive === desired && !options.force) {
    return;
  }

  controlState.pttPending = true;
  syncPttUiState();
  try {
    const state = await RVC.api("/api/v1/realtime/ptt", {
      method: "POST",
      body: JSON.stringify({ active: desired }),
    });
    controlState.pttActive = Boolean(state.ptt_active);
    controlState.realtimeRunning = Boolean(state.running);
    if (state.settings) {
      controlState.currentSettings.ptt_enabled = Boolean(state.settings.ptt_enabled);
    }
    if (RVC.byId("rtState")) {
      RVC.byId("rtState").textContent = JSON.stringify(state, null, 2);
    }
  } catch (err) {
    RVC.logToBox("eventLog", `PTT 更新失败: ${err.message || err}`);
  } finally {
    controlState.pttPending = false;
    syncPttUiState();
  }
}

async function refreshLoadedModels(options = {}) {
  const silent = Boolean(options.silent);
  try {
    await RVC.loadLoadedModels({
      containerId: "loadedModelsStrip",
      logBoxId: "eventLog",
      errorId: "rtError",
      allowUnload: true,
      onUnloaded: () => {
        RVC.loadHealth("healthBadge").catch(() => {});
      },
    });
  } catch (err) {
    if (!silent) {
      RVC.logToBox("eventLog", `模型状态刷新失败: ${err.message || err}`);
    }
  }
}

function startLoadedModelsPolling() {
  if (modelStripPollTimer != null) return;
  modelStripPollTimer = window.setInterval(() => {
    refreshLoadedModels({ silent: true }).catch(() => {});
  }, MODEL_STRIP_POLL_MS);
}

function stopLoadedModelsPolling() {
  if (modelStripPollTimer == null) return;
  window.clearInterval(modelStripPollTimer);
  modelStripPollTimer = null;
}

function syncTuningControlsFromSettings() {
  const inputGain = RVC.byId("inputGainDb");
  const outputGain = RVC.byId("outputGainDb");
  const pttEnabled = RVC.byId("pttEnabled");
  const llmCorrectionEnabled = RVC.byId("llmCorrectionEnabled");
  const denoiseEnabled = RVC.byId("denoiseEnabled");
  const denoiseStrength = RVC.byId("denoiseStrength");
  const ambienceEnabled = RVC.byId("ambienceEnabled");
  const ambienceLevel = RVC.byId("ambienceLevel");

  if (inputGain) inputGain.value = String(toNumber(controlState.currentSettings.input_gain_db, 0));
  if (outputGain) outputGain.value = String(toNumber(controlState.currentSettings.output_gain_db, 0));
  if (pttEnabled) pttEnabled.checked = Boolean(controlState.currentSettings.ptt_enabled);
  if (llmCorrectionEnabled) {
    llmCorrectionEnabled.checked = Boolean(controlState.currentSettings.llm_correction_enabled);
  }
  if (denoiseEnabled) denoiseEnabled.checked = Boolean(controlState.currentSettings.denoise_enabled);
  if (denoiseStrength) {
    denoiseStrength.value = String(toNumber(controlState.currentSettings.denoise_strength, 0.35));
    denoiseStrength.disabled = !(denoiseEnabled?.checked);
  }
  if (ambienceEnabled) ambienceEnabled.checked = Boolean(controlState.currentSettings.ambience_enabled);
  if (ambienceLevel) {
    ambienceLevel.value = String(toNumber(controlState.currentSettings.ambience_level, 0.003));
    ambienceLevel.disabled = !(ambienceEnabled?.checked);
  }
  renderTuningLabels();
  syncPttUiState();
}

function syncTuningSettingsFromControls() {
  controlState.currentSettings.input_gain_db = toNumber(RVC.byId("inputGainDb")?.value, 0);
  controlState.currentSettings.output_gain_db = toNumber(RVC.byId("outputGainDb")?.value, 0);
  controlState.currentSettings.ptt_enabled = Boolean(RVC.byId("pttEnabled")?.checked);
  controlState.currentSettings.llm_correction_enabled = Boolean(
    RVC.byId("llmCorrectionEnabled")?.checked
  );
  controlState.currentSettings.denoise_enabled = Boolean(RVC.byId("denoiseEnabled")?.checked);
  controlState.currentSettings.denoise_strength = toNumber(RVC.byId("denoiseStrength")?.value, 0.35);
  controlState.currentSettings.ambience_enabled = Boolean(RVC.byId("ambienceEnabled")?.checked);
  controlState.currentSettings.ambience_level = toNumber(RVC.byId("ambienceLevel")?.value, 0.003);
  if (!controlState.currentSettings.ptt_enabled) {
    controlState.pttActive = true;
  } else if (!controlState.realtimeRunning) {
    controlState.pttActive = false;
  }
  syncPttUiState();
}

function requireVoiceSelection() {
  const voiceId = RVC.byId("voiceSelect").value;
  if (!voiceId) {
    const message = "请先选择音色，再启动实时链路。";
    RVC.setFormError("rtError", message);
    throw new Error(message);
  }
  return voiceId;
}

function syncRealtimeLanguageFromVoice() {
  const select = RVC.byId("voiceSelect");
  const langSelect = RVC.byId("rtLanguage");
  if (!select || !langSelect) return;
  const voiceId = select.value;
  if (!voiceId) return;
  const voice = (RVC.state.voices || []).find((item) => item.id === voiceId);
  if (!voice) return;
  const hint = String(voice.language_hint || "").trim();
  if (hint === "Chinese" || hint === "English") {
    langSelect.value = hint;
  }
}

async function loadAllControlData() {
  await RVC.loadHealth("healthBadge");
  refreshLoadedModels({ silent: true }).catch(() => {});
  await RVC.loadDevices({
    inputSelectId: "inputDevice",
    virtualSelectId: "virtualDevice",
    monitorSelectId: "monitorDevice",
    deviceNoticeId: "deviceNotice",
  });
  await RVC.loadVoices("voiceSelect");
  syncRealtimeLanguageFromVoice();

  const rtState = await RVC.loadRealtimeState({
    monitorEnabledId: "monitorEnabled",
    inputSelectId: "inputDevice",
    monitorSelectId: "monitorDevice",
    virtualSelectId: "virtualDevice",
    voiceSelectId: "voiceSelect",
    stateBoxId: "rtState",
  });

  controlState.currentSettings.vad_silence_ms = Number(rtState.settings.vad_silence_ms || 140);
  controlState.currentSettings.max_segment_ms = Number(rtState.settings.max_segment_ms || 2800);
  controlState.currentSettings.ptt_enabled = Boolean(rtState.settings.ptt_enabled ?? true);
  controlState.currentSettings.llm_correction_enabled = Boolean(
    rtState.settings.llm_correction_enabled ?? true
  );
  controlState.currentSettings.input_gain_db = Number(rtState.settings.input_gain_db || 0);
  controlState.currentSettings.output_gain_db = Number(rtState.settings.output_gain_db || 0);
  controlState.currentSettings.denoise_enabled = Boolean(rtState.settings.denoise_enabled);
  controlState.currentSettings.denoise_strength = Number(rtState.settings.denoise_strength ?? 0.35);
  controlState.currentSettings.ambience_enabled = Boolean(rtState.settings.ambience_enabled ?? true);
  controlState.currentSettings.ambience_level = Number(rtState.settings.ambience_level ?? 0.003);
  controlState.realtimeRunning = Boolean(rtState.running);
  controlState.pttActive = Boolean(rtState.ptt_active);
  syncTuningControlsFromSettings();

  const metrics = await RVC.api("/api/v1/metrics/current");
  RVC.updateMetrics(metrics);
}

async function applySettings(options = {}) {
  RVC.setFormError("rtError", "");
  syncTuningSettingsFromControls();

  const payload = {
    input_device_id: RVC.selectedNumber("inputDevice"),
    monitor_device_id: RVC.selectedNumber("monitorDevice"),
    virtual_mic_device_id: RVC.selectedNumber("virtualDevice"),
    monitor_enabled: RVC.byId("monitorEnabled").checked,
    ptt_enabled: controlState.currentSettings.ptt_enabled,
    llm_correction_enabled: controlState.currentSettings.llm_correction_enabled,
    vad_silence_ms: controlState.currentSettings.vad_silence_ms,
    max_segment_ms: controlState.currentSettings.max_segment_ms,
    input_gain_db: controlState.currentSettings.input_gain_db,
    output_gain_db: controlState.currentSettings.output_gain_db,
    denoise_enabled: controlState.currentSettings.denoise_enabled,
    denoise_strength: controlState.currentSettings.denoise_strength,
    ambience_enabled: controlState.currentSettings.ambience_enabled,
    ambience_level: controlState.currentSettings.ambience_level,
  };

  await RVC.api("/api/v1/realtime/settings", {
    method: "PATCH",
    body: JSON.stringify(payload),
  });

  const state = await RVC.loadRealtimeState({
    monitorEnabledId: "monitorEnabled",
    inputSelectId: "inputDevice",
    monitorSelectId: "monitorDevice",
    virtualSelectId: "virtualDevice",
    voiceSelectId: "voiceSelect",
    stateBoxId: "rtState",
  });

  controlState.currentSettings.vad_silence_ms = Number(state.settings.vad_silence_ms || 140);
  controlState.currentSettings.max_segment_ms = Number(state.settings.max_segment_ms || 2800);
  controlState.currentSettings.ptt_enabled = Boolean(state.settings.ptt_enabled ?? true);
  controlState.currentSettings.llm_correction_enabled = Boolean(
    state.settings.llm_correction_enabled ?? true
  );
  controlState.currentSettings.input_gain_db = Number(state.settings.input_gain_db || 0);
  controlState.currentSettings.output_gain_db = Number(state.settings.output_gain_db || 0);
  controlState.currentSettings.denoise_enabled = Boolean(state.settings.denoise_enabled);
  controlState.currentSettings.denoise_strength = Number(state.settings.denoise_strength ?? 0.35);
  controlState.currentSettings.ambience_enabled = Boolean(state.settings.ambience_enabled ?? true);
  controlState.currentSettings.ambience_level = Number(state.settings.ambience_level ?? 0.003);
  controlState.realtimeRunning = Boolean(state.running);
  controlState.pttActive = Boolean(state.ptt_active);
  syncTuningControlsFromSettings();

  if (!options.silent) {
    RVC.logToBox("eventLog", "实时设置已更新。")
  }
}

function scheduleSettingsAutosave() {
  if (controlState.settingsAutosaveTimer != null) {
    window.clearTimeout(controlState.settingsAutosaveTimer);
  }
  controlState.settingsAutosaveTimer = window.setTimeout(async () => {
    controlState.settingsAutosaveTimer = null;
    if (controlState.settingsAutosaveInFlight) return;
    controlState.settingsAutosaveInFlight = true;
    try {
      await applySettings({ silent: true });
    } catch (err) {
      RVC.logToBox("eventLog", `自动保存设置失败: ${err.message}`);
    } finally {
      controlState.settingsAutosaveInFlight = false;
    }
  }, 420);
}

async function startRealtime() {
  RVC.setFormError("rtError", "");
  await applySettings({ silent: true });
  const voiceId = requireVoiceSelection();
  resetAudioLevels();

  const payload = {
    voice_id: voiceId,
    language: RVC.byId("rtLanguage").value,
  };

  const data = await RVC.api("/api/v1/realtime/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  if (data.settings?.input_device_id != null) {
    RVC.byId("inputDevice").value = String(data.settings.input_device_id);
  }
  if (data.settings?.virtual_mic_device_id != null) {
    RVC.byId("virtualDevice").value = String(data.settings.virtual_mic_device_id);
  }
  if (data.settings?.monitor_device_id != null) {
    RVC.byId("monitorDevice").value = String(data.settings.monitor_device_id);
  }

  controlState.realtimeRunning = Boolean(data.running);
  controlState.pttActive = Boolean(data.ptt_active);
  syncPttUiState();
  RVC.byId("rtState").textContent = JSON.stringify(data, null, 2);
  RVC.logToBox("eventLog", `实时链路已启动，音色=${voiceId}。`);
  await refreshLoadedModels({ silent: true });
}

async function stopRealtime() {
  RVC.setFormError("rtError", "");
  const data = await RVC.api("/api/v1/realtime/stop", { method: "POST" });
  controlState.realtimeRunning = Boolean(data.running);
  controlState.pttActive = Boolean(data.ptt_active);
  syncPttUiState();
  RVC.byId("rtState").textContent = JSON.stringify(data, null, 2);
  resetAudioLevels();
  RVC.logToBox("eventLog", "实时链路已停止。");
  await refreshLoadedModels({ silent: true });
}

function wireControlActions() {
  RVC.byId("btnApplySettings").addEventListener("click", () =>
    RVC.runWithButton("btnApplySettings", applySettings).catch((err) => {
      RVC.setFormError("rtError", err.message);
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnStart").addEventListener("click", () =>
    RVC.runWithButton("btnStart", startRealtime).catch((err) => {
      RVC.setFormError("rtError", err.message);
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnStop").addEventListener("click", () =>
    RVC.runWithButton("btnStop", stopRealtime).catch((err) => {
      RVC.setFormError("rtError", err.message);
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnRefreshVoices").addEventListener("click", () =>
    RVC.runWithButton("btnRefreshVoices", () => RVC.loadVoices("voiceSelect")).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  ["inputGainDb", "outputGainDb", "denoiseStrength", "ambienceLevel"].forEach((id) => {
    const el = RVC.byId(id);
    if (!el) return;
    el.addEventListener("input", () => {
      syncTuningSettingsFromControls();
      renderTuningLabels();
      scheduleSettingsAutosave();
    });
  });

  const denoiseEnabled = RVC.byId("denoiseEnabled");
  if (denoiseEnabled) {
    denoiseEnabled.addEventListener("change", () => {
      const denoiseStrength = RVC.byId("denoiseStrength");
      if (denoiseStrength) {
        denoiseStrength.disabled = !denoiseEnabled.checked;
      }
      syncTuningSettingsFromControls();
      renderTuningLabels();
      scheduleSettingsAutosave();
    });
  }
  const ambienceEnabled = RVC.byId("ambienceEnabled");
  if (ambienceEnabled) {
    ambienceEnabled.addEventListener("change", () => {
      const ambienceLevel = RVC.byId("ambienceLevel");
      if (ambienceLevel) {
        ambienceLevel.disabled = !ambienceEnabled.checked;
      }
      syncTuningSettingsFromControls();
      renderTuningLabels();
      scheduleSettingsAutosave();
    });
  }

  ["pttEnabled", "llmCorrectionEnabled"].forEach((id) => {
    const el = RVC.byId(id);
    if (!el) return;
    el.addEventListener("change", () => {
      syncTuningSettingsFromControls();
      scheduleSettingsAutosave();
    });
  });

  ["inputDevice", "virtualDevice", "monitorDevice", "monitorEnabled"].forEach((id) => {
    const el = RVC.byId(id);
    if (!el) return;
    el.addEventListener("change", () => {
      scheduleSettingsAutosave();
    });
  });

  const pttBtn = RVC.byId("btnPttHold");
  if (pttBtn) {
    const press = (evt) => {
      evt.preventDefault();
      setPttActive(true).catch(() => {});
    };
    const release = (evt) => {
      evt.preventDefault();
      setPttActive(false).catch(() => {});
    };
    pttBtn.addEventListener("pointerdown", press);
    pttBtn.addEventListener("pointerup", release);
    pttBtn.addEventListener("pointerleave", release);
    pttBtn.addEventListener("pointercancel", release);
  }

  window.addEventListener("keydown", (evt) => {
    if (evt.code !== "Space" || evt.repeat) return;
    if (isEditableElement(evt.target)) return;
    evt.preventDefault();
    setPttActive(true).catch(() => {});
  });
  window.addEventListener("keyup", (evt) => {
    if (evt.code !== "Space") return;
    if (isEditableElement(evt.target)) return;
    evt.preventDefault();
    setPttActive(false).catch(() => {});
  });
  window.addEventListener("blur", () => {
    setPttActive(false).catch(() => {});
  });

  RVC.byId("voiceSelect").addEventListener("change", () => {
    syncRealtimeLanguageFromVoice();
  });
}

function wireRealtimeWs() {
  RVC.connectWs({
    wsBadgeId: "wsBadge",
    onMessage: (msg) => {
      if (msg.type === "latency_tick") {
        RVC.updateMetrics(msg.data);
      } else if (msg.type === "audio_level") {
        updateAudioLevels(msg.data || {});
      } else if (msg.type === "asr_partial") {
        const partialText = String(msg.data?.text || "").trim();
        if (partialText.length >= 4) {
          RVC.logToBox("eventLog", `ASR 临时结果: ${partialText}`);
        }
      } else if (msg.type === "asr_final") {
        RVC.logToBox("eventLog", `ASR 最终结果: ${msg.data.text}`);
      } else if (msg.type === "asr_corrected") {
        RVC.logToBox("eventLog", `纠错后文本: ${msg.data.text}`);
      } else if (msg.type === "error") {
        RVC.logToBox("eventLog", `错误: ${msg.data.message}`);
      } else if (msg.type === "notice") {
        RVC.logToBox("eventLog", `提示: ${msg.data.message}`);
      } else if (msg.type === "ptt_state") {
        controlState.pttActive = Boolean(msg.data?.active);
        if (typeof msg.data?.enabled === "boolean") {
          controlState.currentSettings.ptt_enabled = Boolean(msg.data.enabled);
        }
        syncPttUiState();
      } else if (msg.type === "state_changed") {
        controlState.realtimeRunning = Boolean(msg.data?.running);
        if (typeof msg.data?.ptt_active === "boolean") {
          controlState.pttActive = Boolean(msg.data.ptt_active);
        } else if (!controlState.realtimeRunning) {
          controlState.pttActive = false;
        }
        syncPttUiState();
        RVC.byId("rtState").textContent = JSON.stringify(msg.data, null, 2);
      }
    },
  });
}

async function initControlPage() {
  wireControlActions();
  wireRealtimeWs();
  resetAudioLevels();
  renderTuningLabels();
  syncPttUiState();
  startAudioLevelDecay();
  startLoadedModelsPolling();
  window.addEventListener("focus", () => {
    refreshLoadedModels({ silent: true }).catch(() => {});
  });
  window.addEventListener("beforeunload", () => {
    stopAudioLevelDecay();
    stopLoadedModelsPolling();
  });
  await loadAllControlData();
}

initControlPage().catch((err) => {
  RVC.setFormError("rtError", err.message);
  RVC.logToBox("eventLog", err.message);
});

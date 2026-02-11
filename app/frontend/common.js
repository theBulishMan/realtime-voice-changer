const RVC = (() => {
  const state = {
    voices: [],
    devices: [],
    virtualMicOutputId: null,
    loadedModels: [],
  };

  const LIMITS = {
    designNameMax: 64,
    designPromptMax: 2048,
    designPreviewTextMax: 1024,
    previewTextMax: 5000,
    cloneNameMax: 64,
    cloneFileMaxBytes: 10 * 1024 * 1024,
    cloneAllowedSuffix: new Set([".wav", ".mp3", ".m4a"]),
  };

  const GATE_TARGETS = {
    p95FadMs: 250,
    p95E2eMs: 800,
  };

  const PROJECT_VIRTUAL_MIC_LABEL = "实时变声器虚拟麦";

  function byId(id) {
    return document.getElementById(id);
  }

  function exists(id) {
    return byId(id) !== null;
  }

  function setBadge(el, text, ok) {
    if (!el) return;
    el.textContent = text;
    el.classList.remove("pill-ok", "pill-warn");
    el.classList.add(ok ? "pill-ok" : "pill-warn");
  }

  function setFormError(id, message) {
    const el = byId(id);
    if (!el) return;
    el.textContent = message || "";
  }

  function setNotice(id, message) {
    const el = byId(id);
    if (!el) return;
    if (!message) {
      el.textContent = "";
      el.classList.add("hidden");
      return;
    }
    el.textContent = message;
    el.classList.remove("hidden");
  }

  function logToBox(boxId, line, maxLines = 220) {
    const box = byId(boxId);
    if (!box) return;
    const ts = new Date().toLocaleTimeString();
    const incoming = `[${ts}] ${line}`;
    const current = box.textContent ? box.textContent.split("\n") : [];
    const merged = [incoming, ...current].filter((item) => item.trim().length > 0).slice(0, maxLines);
    box.textContent = merged.join("\n");
    box.scrollTop = 0;
  }

  function parseSuffix(filename) {
    const idx = filename.lastIndexOf(".");
    return idx < 0 ? "" : filename.slice(idx).toLowerCase();
  }

  async function parseApiError(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        const payload = await response.json();
        if (typeof payload?.detail === "string" && payload.detail.trim()) {
          return payload.detail.trim();
        }
        return JSON.stringify(payload);
      } catch (_err) {
        return `HTTP ${response.status}`;
      }
    }

    const text = (await response.text()).trim();
    if (text) {
      return text;
    }
    return `HTTP ${response.status}`;
  }

  async function api(path, options = {}) {
    const method = options.method || "GET";
    const headers = { ...(options.headers || {}) };
    const hasBody = Object.prototype.hasOwnProperty.call(options, "body");

    if (hasBody && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(path, { ...options, method, headers });
    if (!response.ok) {
      throw new Error(await parseApiError(response));
    }
    return response.json();
  }

  async function runWithButton(buttonId, fn) {
    const button = byId(buttonId);
    if (!button) {
      return fn();
    }
    button.disabled = true;
    try {
      return await fn();
    } finally {
      button.disabled = false;
    }
  }

  function selectedNumber(selectId) {
    const el = byId(selectId);
    if (!el) return null;
    const value = el.value;
    if (value === "" || value == null) {
      return null;
    }
    return Number(value);
  }

  function buildDeviceOptions(selectEl, devices, channelKey, role = "input") {
    if (!selectEl) return;

    selectEl.innerHTML = "";
    const valid = devices.filter((d) => Number(d[channelKey] || 0) > 0);

    valid.forEach((device) => {
      const opt = document.createElement("option");
      opt.value = String(device.id);

      let label = device.name;
      if (role !== "input" && state.virtualMicOutputId != null && device.id === state.virtualMicOutputId) {
        label = `${PROJECT_VIRTUAL_MIC_LABEL} (${device.name})`;
      }
      if (channelKey === "max_input_channels" && device.is_default_input) {
        label = `${label} [默认]`;
      }
      if (channelKey === "max_output_channels" && device.is_default_output) {
        label = `${label} [默认]`;
      }

      opt.textContent = label;
      selectEl.appendChild(opt);
    });

    if (valid.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "无可用设备";
      selectEl.appendChild(opt);
    }
  }

  async function loadHealth(badgeId = "healthBadge") {
    const data = await api("/api/v1/health");
    const statusText = data.status === "ok" ? "正常" : (data.status === "degraded" ? "降级" : data.status);
    const modeText = data.fake_mode ? "假数据" : "真实";
    const gpuText = data.gpu_required
      ? (data.gpu_available ? `GPU已就绪(${data.gpu_device || "CUDA"})` : "GPU不可用")
      : "GPU未要求";
    const customText = data.custom_voice_enabled
      ? (data.custom_voice_model_ready ? "CustomVoice可用" : "CustomVoice未就绪")
      : "CustomVoice已禁用";
    setBadge(
      byId(badgeId),
      `健康: ${statusText} / 模式=${modeText} / ${gpuText} / ${customText}`,
      data.status === "ok"
    );
    return data;
  }

  async function loadDevices(options = {}) {
    const data = await api("/api/v1/audio/devices");
    state.devices = data.devices;
    state.virtualMicOutputId = data.virtual_mic_output_id ?? null;

    const inputSelect = options.inputSelectId ? byId(options.inputSelectId) : null;
    const virtualSelect = options.virtualSelectId ? byId(options.virtualSelectId) : null;
    const monitorSelect = options.monitorSelectId ? byId(options.monitorSelectId) : null;

    buildDeviceOptions(inputSelect, state.devices, "max_input_channels", "input");
    buildDeviceOptions(virtualSelect, state.devices, "max_output_channels", "virtual");
    buildDeviceOptions(monitorSelect, state.devices, "max_output_channels", "monitor");

    if (inputSelect && data.default_input_id != null) {
      inputSelect.value = String(data.default_input_id);
    }
    if (monitorSelect && data.default_output_id != null) {
      monitorSelect.value = String(data.default_output_id);
    }
    if (virtualSelect && data.virtual_mic_output_id != null) {
      virtualSelect.value = String(data.virtual_mic_output_id);
    }

    if (options.deviceNoticeId) {
      if (data.virtual_mic_output_id != null) {
        const found = state.devices.find((item) => item.id === data.virtual_mic_output_id);
        const name = found ? found.name : "CABLE Input";
        setNotice(options.deviceNoticeId, `虚拟麦已就绪: ${PROJECT_VIRTUAL_MIC_LABEL} (${name})`);
      } else {
        setNotice(
          options.deviceNoticeId,
          "未检测到虚拟麦。请用管理员权限运行 scripts/install_vbcable.ps1，然后重启后端。"
        );
      }
    }

    return data;
  }

  async function loadVoices(selectId = "voiceSelect") {
    const voices = await api("/api/v1/voices");
    state.voices = voices;
    const sel = byId(selectId);
    if (sel) {
      const previous = sel.value;
      sel.innerHTML = "";
      voices.forEach((voice) => {
        const opt = document.createElement("option");
        opt.value = voice.id;
        opt.textContent = `${voice.name} (${voice.mode})`;
        sel.appendChild(opt);
      });
      if (voices.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "暂无音色";
        sel.appendChild(opt);
      } else if (voices.some((voice) => voice.id === previous)) {
        sel.value = previous;
      } else {
        sel.value = voices[0].id;
      }
    }
    return voices;
  }

  function renderLoadedModels(container, data, options = {}) {
    container.innerHTML = "";
    const models = Array.isArray(data.models) ? data.models : [];
    state.loadedModels = models;

    const memoryChip = document.createElement("div");
    memoryChip.className = "model-chip model-chip-summary";
    const allocated = Number(data.cuda_memory_allocated_mb || 0);
    const reserved = Number(data.cuda_memory_reserved_mb || 0);
    memoryChip.textContent = `显存: 已分配 ${allocated.toFixed(1)}MB / 保留 ${reserved.toFixed(1)}MB`;
    container.appendChild(memoryChip);

    if (models.length === 0) {
      const emptyChip = document.createElement("div");
      emptyChip.className = "model-chip";
      emptyChip.textContent = "暂无已加载模型";
      container.appendChild(emptyChip);
      return;
    }

    models.forEach((item) => {
      const chip = document.createElement("div");
      chip.className = "model-chip";

      const text = document.createElement("span");
      const replicas = Math.max(1, Number(item.replicas || 1));
      const suffix = replicas > 1 ? ` ×${replicas}` : "";
      const device = String(item.device || "").trim();
      text.textContent = `${item.label}${suffix}${device ? ` @ ${device}` : ""}`;
      chip.appendChild(text);

      if (options.allowUnload !== false && item.unloadable !== false) {
        const close = document.createElement("button");
        close.type = "button";
        close.className = "model-chip-close";
        close.textContent = "×";
        close.title = `卸载 ${item.label}`;
        close.setAttribute("aria-label", `卸载 ${item.label}`);
        close.addEventListener("click", async (evt) => {
          evt.preventDefault();
          evt.stopPropagation();
          close.disabled = true;
          try {
            await api(`/api/v1/models/loaded/${encodeURIComponent(item.key)}`, {
              method: "DELETE",
            });
            if (options.logBoxId) {
              logToBox(options.logBoxId, `已卸载模型: ${item.label}`);
            }
            await loadLoadedModels(options);
            if (typeof options.onUnloaded === "function") {
              options.onUnloaded(item.key);
            }
          } catch (err) {
            if (options.errorId) {
              setFormError(options.errorId, err.message || String(err));
            }
            if (options.logBoxId) {
              logToBox(options.logBoxId, `卸载失败: ${err.message || err}`);
            }
          } finally {
            close.disabled = false;
          }
        });
        chip.appendChild(close);
      }

      container.appendChild(chip);
    });
  }

  async function loadLoadedModels(options = {}) {
    const containerId = options.containerId || "loadedModelsStrip";
    const container = byId(containerId);
    if (!container) return null;
    const data = await api("/api/v1/models/loaded");
    renderLoadedModels(container, data, options);
    return data;
  }

  async function loadRealtimeState(options = {}) {
    const data = await api("/api/v1/realtime/state");

    if (options.monitorEnabledId && exists(options.monitorEnabledId)) {
      byId(options.monitorEnabledId).checked = Boolean(data.settings.monitor_enabled);
    }
    if (options.inputSelectId && exists(options.inputSelectId) && data.settings.input_device_id != null) {
      byId(options.inputSelectId).value = String(data.settings.input_device_id);
    }
    if (options.monitorSelectId && exists(options.monitorSelectId) && data.settings.monitor_device_id != null) {
      byId(options.monitorSelectId).value = String(data.settings.monitor_device_id);
    }
    if (options.virtualSelectId && exists(options.virtualSelectId) && data.settings.virtual_mic_device_id != null) {
      byId(options.virtualSelectId).value = String(data.settings.virtual_mic_device_id);
    }
    if (options.voiceSelectId && exists(options.voiceSelectId) && data.voice_id) {
      byId(options.voiceSelectId).value = data.voice_id;
    }
    if (options.stateBoxId && exists(options.stateBoxId)) {
      byId(options.stateBoxId).textContent = JSON.stringify(data, null, 2);
    }

    return data;
  }

  function setLatencyGates(metrics, ids = {}) {
    const gateFadId = ids.gateFadId || "gateFad";
    const gateE2eId = ids.gateE2eId || "gateE2E";
    const samplesId = ids.samplesId || "metricSamples";

    const sampleCount = Number(metrics.sample_count || 0);
    const samplesEl = byId(samplesId);
    if (samplesEl) {
      samplesEl.textContent = `样本数: ${sampleCount}`;
    }

    if (sampleCount <= 0) {
      setBadge(byId(gateFadId), "首包延迟 P95: 采样中", false);
      setBadge(byId(gateE2eId), "全链路延迟 P95: 采样中", false);
      return;
    }

    const fad = Number(metrics.p95_fad_ms || 0);
    const e2e = Number(metrics.p95_e2e_ms || 0);
    const fadOk = fad <= GATE_TARGETS.p95FadMs;
    const e2eOk = e2e <= GATE_TARGETS.p95E2eMs;

    setBadge(byId(gateFadId), `首包延迟 P95: ${fad.toFixed(1)}ms / 门限 ${GATE_TARGETS.p95FadMs}ms`, fadOk);
    setBadge(byId(gateE2eId), `全链路延迟 P95: ${e2e.toFixed(1)}ms / 门限 ${GATE_TARGETS.p95E2eMs}ms`, e2eOk);
  }

  function updateMetrics(metrics, ids = {}) {
    const p50FadId = ids.p50FadId || "mP50Fad";
    const p95FadId = ids.p95FadId || "mP95Fad";
    const p50E2eId = ids.p50E2eId || "mP50E2E";
    const p95E2eId = ids.p95E2eId || "mP95E2E";

    const p50FadEl = byId(p50FadId);
    const p95FadEl = byId(p95FadId);
    const p50E2eEl = byId(p50E2eId);
    const p95E2eEl = byId(p95E2eId);
    const asrMsEl = byId(ids.asrMsId || "mAsrMs");
    const ttsMsEl = byId(ids.ttsMsId || "mTtsMs");

    if (p50FadEl) p50FadEl.textContent = `${Number(metrics.p50_fad_ms || 0).toFixed(1)}ms`;
    if (p95FadEl) p95FadEl.textContent = `${Number(metrics.p95_fad_ms || 0).toFixed(1)}ms`;
    if (p50E2eEl) p50E2eEl.textContent = `${Number(metrics.p50_e2e_ms || 0).toFixed(1)}ms`;
    if (p95E2eEl) p95E2eEl.textContent = `${Number(metrics.p95_e2e_ms || 0).toFixed(1)}ms`;
    if (asrMsEl) asrMsEl.textContent = `${Number(metrics.asr_ms || 0).toFixed(1)}ms`;
    if (ttsMsEl) ttsMsEl.textContent = `${Number(metrics.tts_ms || 0).toFixed(1)}ms`;

    setLatencyGates(metrics, ids);
  }

  function connectWs(options = {}) {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${window.location.host}/ws/realtime`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      if (options.wsBadgeId) {
        setBadge(byId(options.wsBadgeId), "WS 已连接", true);
      }
      if (typeof options.onOpen === "function") {
        options.onOpen();
      }
    };

    ws.onclose = () => {
      if (options.wsBadgeId) {
        setBadge(byId(options.wsBadgeId), "WS 未连接", false);
      }
      if (typeof options.onClose === "function") {
        options.onClose();
      }
      setTimeout(() => connectWs(options), 1500);
    };

    ws.onerror = () => {
      if (options.wsBadgeId) {
        setBadge(byId(options.wsBadgeId), "WS 错误", false);
      }
      if (typeof options.onError === "function") {
        options.onError();
      }
    };

    ws.onmessage = (evt) => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch (_err) {
        return;
      }
      if (typeof options.onMessage === "function") {
        options.onMessage(msg);
      }
    };

    return ws;
  }

  return {
    state,
    LIMITS,
    GATE_TARGETS,
    PROJECT_VIRTUAL_MIC_LABEL,
    byId,
    setBadge,
    setFormError,
    setNotice,
    logToBox,
    parseSuffix,
    api,
    runWithButton,
    selectedNumber,
    loadHealth,
    loadDevices,
    loadVoices,
    loadLoadedModels,
    loadRealtimeState,
    updateMetrics,
    connectWs,
  };
})();

window.RVC = RVC;

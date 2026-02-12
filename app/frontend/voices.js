const DEFAULT_DESIGN_PROMPT_TEMPLATE =
  "一位自然、真实、贴近人声的说话者，吐字清晰，语速中等偏稳，" +
  "语气友好不过度夸张，情绪克制但有亲和力，句尾收音自然，避免电音感和机械感。";

const DEFAULT_LONG_PREVIEW_TEXT =
  "大家好，欢迎使用实时变声器。现在做一段通用试听：包含短句、长句、停顿和语气变化。" +
  "如果你能清楚听到每个字并且觉得声音自然、不发闷、不刺耳，说明当前音色已经比较稳定。" +
  "接下来我们继续读一段英文 mixed content for robustness check, " +
  "including numbers like one two three and simple punctuation.";

const FALLBACK_CUSTOM_SPEAKERS = [
  { id: "Vivian", label: "薇薇安 (Vivian)", description: "明亮、略带棱角的年轻女声。", native_language: "Chinese" },
  { id: "Serena", label: "塞雷娜 (Serena)", description: "温暖、柔和的年轻女性声音。", native_language: "Chinese" },
  { id: "Uncle_Fu", label: "福叔 (Uncle_Fu)", description: "经验丰富的男声，音色低沉醇厚。", native_language: "Chinese" },
  { id: "Dylan", label: "迪伦 (Dylan)", description: "年轻的北京男声，音色清晰自然。", native_language: "Chinese (Beijing)" },
  { id: "Eric", label: "埃里克 (Eric)", description: "活泼的成都男声，略带沙哑的明亮音色。", native_language: "Chinese (Sichuan)" },
  { id: "Ryan", label: "瑞安 (Ryan)", description: "富有活力、节奏感强的男声。", native_language: "English" },
  { id: "Aiden", label: "艾登 (Aiden)", description: "阳光的美式男声，中音清晰。", native_language: "English" },
  { id: "Ono_Anna", label: "小野安娜 (Ono_Anna)", description: "活泼俏皮的日语女声，音色轻快灵动。", native_language: "Japanese" },
  { id: "Sohee", label: "昭熙 (Sohee)", description: "温暖且富感染力的韩语女声。", native_language: "Korean" },
];

const FALLBACK_CUSTOM_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean"];
const MODEL_STRIP_POLL_MS = 4000;
let modelStripPollTimer = null;

let customCatalog = {
  speakers: [...FALLBACK_CUSTOM_SPEAKERS],
  languages: [...FALLBACK_CUSTOM_LANGUAGES],
};

async function refreshLoadedModels(options = {}) {
  const silent = Boolean(options.silent);
  try {
    await RVC.loadLoadedModels({
      containerId: "loadedModelsStrip",
      logBoxId: "eventLog",
      errorId: "previewError",
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

function pickPreviewText(value) {
  const text = String(value || "").trim();
  return text || DEFAULT_LONG_PREVIEW_TEXT;
}

function fillDesignPromptTemplate() {
  const el = RVC.byId("designPrompt");
  if (!el) return;
  el.value = DEFAULT_DESIGN_PROMPT_TEMPLATE;
}

function fillDefaultPreviewText(targetId) {
  const el = RVC.byId(targetId);
  if (!el) return;
  el.value = DEFAULT_LONG_PREVIEW_TEXT;
}

function applyDefaultVoiceWorkbenchTemplates() {
  const designPrompt = RVC.byId("designPrompt");
  if (designPrompt && !designPrompt.value.trim()) {
    designPrompt.value = DEFAULT_DESIGN_PROMPT_TEMPLATE;
  }
  const designText = RVC.byId("designText");
  if (designText && !designText.value.trim()) {
    designText.value = DEFAULT_LONG_PREVIEW_TEXT;
  }
  const previewText = RVC.byId("previewText");
  if (previewText && !previewText.value.trim()) {
    previewText.value = DEFAULT_LONG_PREVIEW_TEXT;
  }
}

async function aiComposeDesignPrompt() {
  const briefEl = RVC.byId("designAiBrief");
  const langEl = RVC.byId("designLang");
  const source = getCurrentDesignSource();
  const brief = String(briefEl?.value || "").trim();
  const language = String(langEl?.value || "Auto");
  if (!brief) {
    const message = "请先输入你想要的声音描述，再点 AI 生成。";
    RVC.setFormError("designError", message);
    throw new Error(message);
  }
  RVC.setFormError("designError", "");
  if (source === "custom") {
    const speaker = String(RVC.byId("customSpeakerSelect")?.value || "").trim();
    const data = await RVC.api("/api/v1/voices/custom/assist", {
      method: "POST",
      body: JSON.stringify({ brief, language, speaker }),
    });
    if (RVC.byId("customInstruct") && data.instruct) {
      RVC.byId("customInstruct").value = data.instruct;
    }
    if (RVC.byId("designText") && data.preview_text) {
      RVC.byId("designText").value = data.preview_text;
    }
    const resultSource = data.source === "siliconflow" ? "硅基流动" : "本地回退";
    RVC.logToBox("eventLog", `AI 已生成语气指令（来源=${resultSource}，模型=${data.model || "-" }）。`);
    return;
  }

  const data = await RVC.api("/api/v1/voices/design/assist", {
    method: "POST",
    body: JSON.stringify({ brief, language }),
  });
  if (RVC.byId("designPrompt") && data.voice_prompt) {
    RVC.byId("designPrompt").value = data.voice_prompt;
  }
  if (RVC.byId("designText") && data.preview_text) {
    RVC.byId("designText").value = data.preview_text;
  }
  const resultSource = data.source === "siliconflow" ? "硅基流动" : "本地回退";
  RVC.logToBox("eventLog", `AI 编排已生成（来源=${resultSource}，模型=${data.model || "-" }）。`);
}

function languageLabel(language) {
  if (language === "Auto") return "自动";
  if (language === "Chinese") return "中文";
  if (language === "English") return "英文";
  if (language === "Japanese") return "日文";
  if (language === "Korean") return "韩文";
  return language;
}

function renderLanguageOptions(selectId, languages) {
  const select = RVC.byId(selectId);
  if (!select) return;
  const wanted = new Set(["Auto", ...languages.filter((item) => String(item || "").trim())]);
  const previous = select.value || "Auto";
  select.innerHTML = "";
  Array.from(wanted).forEach((lang) => {
    const opt = document.createElement("option");
    opt.value = lang;
    opt.textContent = languageLabel(lang);
    select.appendChild(opt);
  });
  select.value = wanted.has(previous) ? previous : "Auto";
}

function renderCustomSpeakerOptions() {
  const select = RVC.byId("customSpeakerSelect");
  if (!select) return;
  const previous = select.value;
  select.innerHTML = "";

  if (!customCatalog.speakers.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "无可用官方音色";
    select.appendChild(opt);
    return;
  }

  customCatalog.speakers.forEach((speaker) => {
    const opt = document.createElement("option");
    opt.value = speaker.id;
    const desc = speaker.description ? ` - ${speaker.description}` : "";
    opt.textContent = `${speaker.label || speaker.id}${desc}`;
    select.appendChild(opt);
  });

  if (customCatalog.speakers.some((speaker) => speaker.id === previous)) {
    select.value = previous;
  } else {
    select.value = customCatalog.speakers[0].id;
  }
}

function getCurrentDesignSource() {
  return (RVC.byId("designSource")?.value || "design").trim();
}

function toggleDesignSourceUI() {
  const source = getCurrentDesignSource();
  const isCustom = source === "custom";

  const promptGroup = RVC.byId("designPromptGroup");
  const customGroup = RVC.byId("customVoiceGroup");

  if (promptGroup) {
    promptGroup.classList.toggle("hidden", isCustom);
  }
  if (customGroup) {
    customGroup.classList.toggle("hidden", !isCustom);
  }
  const aiBtn = RVC.byId("btnDesignAiCompose");
  if (aiBtn) {
    aiBtn.textContent = isCustom ? "AI 生成语气指令与试听文本" : "AI 生成描述与试听文本";
  }
}

async function loadCustomCatalog() {
  try {
    const data = await RVC.api("/api/v1/voices/custom/catalog");
    const speakers = Array.isArray(data.speakers) ? data.speakers : [];
    const languages = Array.isArray(data.languages) ? data.languages : [];
    if (speakers.length > 0) {
      customCatalog = {
        speakers,
        languages: languages.length > 0 ? languages : [...FALLBACK_CUSTOM_LANGUAGES],
      };
    } else {
      customCatalog = {
        speakers: [...FALLBACK_CUSTOM_SPEAKERS],
        languages: [...FALLBACK_CUSTOM_LANGUAGES],
      };
    }
  } catch (_err) {
    customCatalog = {
      speakers: [...FALLBACK_CUSTOM_SPEAKERS],
      languages: [...FALLBACK_CUSTOM_LANGUAGES],
    };
  }

  renderCustomSpeakerOptions();
  renderLanguageOptions("designLang", customCatalog.languages);
}

function validateDesignPayload() {
  const source = getCurrentDesignSource();
  const name = RVC.byId("designName").value.trim();
  const previewText = pickPreviewText(RVC.byId("designText").value);
  const language = RVC.byId("designLang").value;

  if (!name) return { error: "音色名称不能为空。" };
  if (name.length > RVC.LIMITS.designNameMax) {
    return { error: `音色名称不能超过 ${RVC.LIMITS.designNameMax} 个字符。` };
  }
  if (previewText.length > RVC.LIMITS.designPreviewTextMax) {
    return { error: `试听文本不能超过 ${RVC.LIMITS.designPreviewTextMax} 个字符。` };
  }

  if (source === "custom") {
    const speaker = (RVC.byId("customSpeakerSelect")?.value || "").trim();
    const instruct = (RVC.byId("customInstruct")?.value || "").trim();

    if (!speaker) {
      return { error: "请选择官方说话人。" };
    }
    if (instruct.length > 1024) {
      return { error: "语气指令不能超过 1024 个字符。" };
    }

    return {
      endpoint: "/api/v1/voices/custom",
      source,
      payload: {
        name,
        speaker,
        preview_text: previewText,
        language,
        instruct,
      },
    };
  }

  const voicePrompt = RVC.byId("designPrompt").value.trim();
  if (!voicePrompt) return { error: "音色描述不能为空。" };
  if (voicePrompt.length > RVC.LIMITS.designPromptMax) {
    return { error: `音色描述不能超过 ${RVC.LIMITS.designPromptMax} 个字符。` };
  }

  return {
    endpoint: "/api/v1/voices/design",
    source,
    payload: {
      name,
      voice_prompt: voicePrompt,
      preview_text: previewText,
      language,
    },
  };
}

function validateClonePayload() {
  const name = RVC.byId("cloneName").value.trim();
  const refText = RVC.byId("cloneText").value.trim();
  const language = RVC.byId("cloneLang").value;
  const file = RVC.byId("cloneFile").files[0];

  if (!name) return { error: "克隆音色名称不能为空。" };
  if (name.length > RVC.LIMITS.cloneNameMax) {
    return { error: `克隆音色名称不能超过 ${RVC.LIMITS.cloneNameMax} 个字符。` };
  }
  if (!refText) return { error: "参考文本不能为空。" };
  if (!file) return { error: "请选择参考音频文件。" };

  const suffix = RVC.parseSuffix(file.name || "");
  if (!RVC.LIMITS.cloneAllowedSuffix.has(suffix)) {
    return { error: "参考音频格式必须是 WAV / MP3 / M4A。" };
  }
  if (file.size > RVC.LIMITS.cloneFileMaxBytes) {
    return { error: "参考音频文件过大（最大 10MB）。" };
  }

  return {
    payload: {
      name,
      refText,
      language,
      file,
    },
  };
}

async function createDesignVoice(save) {
  const check = validateDesignPayload();
  if (check.error) {
    RVC.setFormError("designError", check.error);
    throw new Error(check.error);
  }
  RVC.setFormError("designError", "");

  const data = await RVC.api(check.endpoint, {
    method: "POST",
    body: JSON.stringify({ ...check.payload, save }),
  });

  RVC.byId("audioPreview").src = `data:audio/wav;base64,${data.preview_audio_b64}`;

  if (save) {
    await RVC.loadVoices("voiceSelect");
    RVC.byId("voiceSelect").value = data.voice.id;
    RVC.logToBox("eventLog", `音色已固化并保存: ${data.voice.name}（来源=${check.source}）`);
  } else {
    RVC.logToBox("eventLog", `已生成试听，可直接固化保存（来源=${check.source}）。`);
  }
}

async function createCloneVoice() {
  const check = validateClonePayload();
  if (check.error) {
    RVC.setFormError("cloneError", check.error);
    throw new Error(check.error);
  }
  RVC.setFormError("cloneError", "");

  const form = new FormData();
  form.append("name", check.payload.name);
  form.append("language", check.payload.language);
  form.append("ref_text", check.payload.refText);
  form.append("audio_file", check.payload.file);

  const response = await fetch("/api/v1/voices/clone", {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const message = await response.text();
    const detail = message || `HTTP ${response.status}`;
    RVC.setFormError("cloneError", detail);
    throw new Error(detail);
  }

  const data = await response.json();
  RVC.byId("audioPreview").src = `data:audio/wav;base64,${data.preview_audio_b64}`;
  await RVC.loadVoices("voiceSelect");
  RVC.byId("voiceSelect").value = data.voice.id;
  RVC.logToBox("eventLog", `克隆音色已保存: ${data.voice.name}`);
}

async function previewVoice() {
  const voiceId = RVC.byId("voiceSelect").value;
  const text = pickPreviewText(RVC.byId("previewText").value);

  if (!voiceId) {
    const message = "请先选择要试听的音色。";
    RVC.setFormError("previewError", message);
    throw new Error(message);
  }
  if (text.length > RVC.LIMITS.previewTextMax) {
    const message = `试听文本不能超过 ${RVC.LIMITS.previewTextMax} 个字符。`;
    RVC.setFormError("previewError", message);
    throw new Error(message);
  }
  RVC.setFormError("previewError", "");

  const data = await RVC.api(`/api/v1/voices/${voiceId}/preview`, {
    method: "POST",
    body: JSON.stringify({ text, language: "Auto" }),
  });
  RVC.byId("audioPreview").src = `data:audio/wav;base64,${data.audio_b64}`;
  RVC.logToBox("eventLog", `已生成试听音频，音色=${voiceId}。`);
}

async function deleteVoice() {
  const voiceId = RVC.byId("voiceSelect").value;
  if (!voiceId) {
    const message = "请先选择要删除的音色。";
    RVC.setFormError("previewError", message);
    throw new Error(message);
  }
  const confirmed = window.confirm("确认删除当前音色？此操作不可恢复。");
  if (!confirmed) {
    return;
  }

  await RVC.api(`/api/v1/voices/${voiceId}`, { method: "DELETE" });
  const voices = await RVC.loadVoices("voiceSelect");
  if (voices.length === 0) {
    RVC.byId("audioPreview").removeAttribute("src");
  }
  RVC.setFormError("previewError", "");
  RVC.logToBox("eventLog", `音色已删除: ${voiceId}`);
}

function setTopDeviceName(id, text) {
  const el = RVC.byId(id);
  if (!el) return;
  el.textContent = text;
}

function resolveDeviceName(deviceId, channelKey) {
  if (deviceId == null) {
    return null;
  }
  const target = Number(deviceId);
  const hit = RVC.state.devices.find(
    (item) => item.id === target && Number(item[channelKey] || 0) > 0
  );
  return hit ? hit.name : null;
}

function resolveFallbackDeviceName(channelKey) {
  const hit = RVC.state.devices.find((item) => Number(item[channelKey] || 0) > 0);
  return hit ? hit.name : null;
}

function renderTopDevices(audioDevices, realtimeState) {
  const inputId = realtimeState.settings?.input_device_id ?? audioDevices.default_input_id ?? null;
  const monitorId = realtimeState.settings?.monitor_device_id ?? audioDevices.default_output_id ?? null;
  const virtualId =
    realtimeState.settings?.virtual_mic_device_id ?? audioDevices.virtual_mic_output_id ?? null;

  const inputName =
    resolveDeviceName(inputId, "max_input_channels") ||
    resolveFallbackDeviceName("max_input_channels") ||
    "未选择";
  const speakerName =
    resolveDeviceName(monitorId, "max_output_channels") ||
    resolveFallbackDeviceName("max_output_channels") ||
    "未选择";
  const virtualName =
    resolveDeviceName(virtualId, "max_output_channels") ||
    (audioDevices.virtual_mic_output_id != null
      ? resolveDeviceName(audioDevices.virtual_mic_output_id, "max_output_channels")
      : null) ||
    "未检测到";

  setTopDeviceName("topInputDevice", inputName);
  setTopDeviceName("topSpeakerDevice", speakerName);
  setTopDeviceName("topVirtualMicDevice", virtualName);
}

async function refreshTopDevices() {
  const audioDevices = await RVC.loadDevices();
  const realtimeState = await RVC.loadRealtimeState();
  renderTopDevices(audioDevices, realtimeState);
}

function wireVoiceActions() {
  RVC.byId("designSource").addEventListener("change", () => {
    toggleDesignSourceUI();
  });

  RVC.byId("btnRefreshCustomCatalog").addEventListener("click", () =>
    RVC.runWithButton("btnRefreshCustomCatalog", async () => {
      await loadCustomCatalog();
      RVC.logToBox("eventLog", "官方音色目录已刷新。");
    }).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnDesignFillPrompt").addEventListener("click", () => {
    fillDesignPromptTemplate();
    RVC.logToBox("eventLog", "已填入通用音色描述模板。");
  });

  RVC.byId("btnDesignFillText").addEventListener("click", () => {
    fillDefaultPreviewText("designText");
    RVC.logToBox("eventLog", "已填入默认长试听文本。");
  });

  RVC.byId("btnCustomFillText").addEventListener("click", () => {
    fillDefaultPreviewText("designText");
    RVC.logToBox("eventLog", "已填入默认长试听文本。");
  });

  RVC.byId("btnPreviewFillText").addEventListener("click", () => {
    fillDefaultPreviewText("previewText");
    RVC.logToBox("eventLog", "试听文本已切换为默认长文本。");
  });

  RVC.byId("btnDesignAiCompose").addEventListener("click", () =>
    RVC.runWithButton("btnDesignAiCompose", aiComposeDesignPrompt).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnDesignPreview").addEventListener("click", () =>
    RVC.runWithButton("btnDesignPreview", () => createDesignVoice(false)).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnDesignCreate").addEventListener("click", () =>
    RVC.runWithButton("btnDesignCreate", () => createDesignVoice(true)).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnCloneCreate").addEventListener("click", () =>
    RVC.runWithButton("btnCloneCreate", createCloneVoice).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnPreview").addEventListener("click", () =>
    RVC.runWithButton("btnPreview", previewVoice).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnRefreshVoices").addEventListener("click", () =>
    RVC.runWithButton("btnRefreshVoices", () => RVC.loadVoices("voiceSelect")).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );

  RVC.byId("btnDeleteVoice").addEventListener("click", () =>
    RVC.runWithButton("btnDeleteVoice", deleteVoice).catch((err) => {
      RVC.logToBox("eventLog", err.message);
    })
  );
}

async function initVoicePage() {
  applyDefaultVoiceWorkbenchTemplates();
  toggleDesignSourceUI();
  wireVoiceActions();
  await loadCustomCatalog();
  await RVC.loadHealth("healthBadge");
  await refreshLoadedModels({ silent: true });
  await RVC.loadVoices("voiceSelect");
  await refreshTopDevices();
  startLoadedModelsPolling();

  window.addEventListener("focus", () => {
    refreshTopDevices().catch((err) => {
      RVC.logToBox("eventLog", err.message);
    });
    refreshLoadedModels({ silent: true }).catch(() => {});
  });

  window.addEventListener("beforeunload", () => {
    stopLoadedModelsPolling();
  });
}

initVoicePage().catch((err) => {
  RVC.logToBox("eventLog", err.message);
});

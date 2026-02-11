const DEFAULT_SF_BASE = "https://api.siliconflow.cn/v1";
const DEFAULT_SF_MODEL = "zai-org/GLM-4.5-Air";

function updateSfStatus(data) {
  const statusEl = RVC.byId("sfStatus");
  if (!statusEl || !data) return;
  const provider = data.text_correction_provider === "siliconflow" ? "远端 API" : "本地模型";
  const displayName = String(data.model_display_name || "未命名模型");
  const keyStatus = data.api_key_present
    ? `已保存 (${data.api_key_masked || "已隐藏"})`
    : "未保存";
  statusEl.textContent = `AI 配置: ${displayName} / 提供方: ${provider} / Key: ${keyStatus}`;
}

function applySiliconFlowSettingsForm(data) {
  if (!data) return;
  const providerEl = RVC.byId("sfProvider");
  const credentialNameEl = RVC.byId("sfCredentialName");
  const modelDisplayNameEl = RVC.byId("sfModelDisplayName");
  const modelEl = RVC.byId("sfModel");
  const baseEl = RVC.byId("sfBase");
  const contextLengthEl = RVC.byId("sfContextLength");
  const maxTokensEl = RVC.byId("sfMaxTokens");
  const compatModeEl = RVC.byId("sfCompatMode");
  const thinkingEnabledEl = RVC.byId("sfThinkingEnabled");
  const timeoutEl = RVC.byId("sfTimeout");
  const keyEl = RVC.byId("sfApiKey");

  if (providerEl) providerEl.value = data.text_correction_provider || "siliconflow";
  if (credentialNameEl) credentialNameEl.value = data.credential_name || "API KEY 1";
  if (modelDisplayNameEl) {
    modelDisplayNameEl.value = data.model_display_name || "远端纠错模型";
  }
  if (modelEl) modelEl.value = data.endpoint_model_name || data.siliconflow_model || DEFAULT_SF_MODEL;
  if (baseEl) baseEl.value = data.siliconflow_api_base || DEFAULT_SF_BASE;
  if (contextLengthEl) {
    contextLengthEl.value = String(Number(data.context_length || 131072));
  }
  if (maxTokensEl) {
    maxTokensEl.value = String(Number(data.max_tokens || 4096));
  }
  if (compatModeEl) {
    compatModeEl.value = data.compat_mode || "strict_openai";
  }
  if (thinkingEnabledEl) {
    thinkingEnabledEl.checked = Boolean(data.thinking_enabled);
  }
  if (timeoutEl) timeoutEl.value = String(Number(data.siliconflow_timeout_s || 12));
  if (keyEl) keyEl.value = "";

  updateSfStatus(data);
  RVC.setFormError("sfError", "");
}

async function loadSiliconFlowSettings() {
  const data = await RVC.api("/api/v1/settings/siliconflow");
  applySiliconFlowSettingsForm(data);
  return data;
}

function buildSiliconFlowSavePayload(options = {}) {
  const clearKey = Boolean(options.clearKey);
  const provider = (RVC.byId("sfProvider")?.value || "siliconflow").trim();
  const credentialName = (RVC.byId("sfCredentialName")?.value || "API KEY 1").trim();
  const modelDisplayName = (RVC.byId("sfModelDisplayName")?.value || "远端纠错模型").trim();
  const model = (RVC.byId("sfModel")?.value || DEFAULT_SF_MODEL).trim();
  const base = (RVC.byId("sfBase")?.value || DEFAULT_SF_BASE).trim();
  const contextRaw = Number(RVC.byId("sfContextLength")?.value || 131072);
  const contextLength = Number.isFinite(contextRaw)
    ? Math.min(262144, Math.max(2048, Math.round(contextRaw)))
    : 131072;
  const maxTokensRaw = Number(RVC.byId("sfMaxTokens")?.value || 4096);
  const maxTokens = Number.isFinite(maxTokensRaw)
    ? Math.min(131072, Math.max(64, Math.round(maxTokensRaw)))
    : 4096;
  const compatMode = (RVC.byId("sfCompatMode")?.value || "strict_openai").trim();
  const thinkingEnabled = Boolean(RVC.byId("sfThinkingEnabled")?.checked);
  const timeoutRaw = Number(RVC.byId("sfTimeout")?.value || 12);
  const timeout = Number.isFinite(timeoutRaw) ? Math.min(120, Math.max(2, timeoutRaw)) : 12;

  const payload = {
    text_correction_provider: provider === "local" ? "local" : "siliconflow",
    credential_name: credentialName || "API KEY 1",
    model_display_name: modelDisplayName || "远端纠错模型",
    endpoint_model_name: model || DEFAULT_SF_MODEL,
    siliconflow_model: model || DEFAULT_SF_MODEL,
    siliconflow_api_base: base || DEFAULT_SF_BASE,
    context_length: contextLength,
    max_tokens: maxTokens,
    compat_mode: compatMode === "siliconflow" ? "siliconflow" : "strict_openai",
    thinking_enabled: thinkingEnabled,
    siliconflow_timeout_s: timeout,
  };

  if (clearKey) {
    payload.clear_api_key = true;
  } else {
    const key = String(RVC.byId("sfApiKey")?.value || "").trim();
    if (key) {
      payload.siliconflow_api_key = key;
    }
  }

  return payload;
}

async function saveSiliconFlowSettings(options = {}) {
  RVC.setFormError("sfError", "");
  const clearKey = Boolean(options.clearKey);
  const payload = buildSiliconFlowSavePayload({ clearKey });
  const data = await RVC.api("/api/v1/settings/siliconflow", {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  applySiliconFlowSettingsForm(data);
}

function wireSettingsActions() {
  RVC.byId("btnSfReload").addEventListener("click", () =>
    RVC.runWithButton("btnSfReload", loadSiliconFlowSettings).catch((err) => {
      RVC.setFormError("sfError", err.message);
    })
  );

  RVC.byId("btnSfSave").addEventListener("click", () =>
    RVC.runWithButton("btnSfSave", async () => {
      await saveSiliconFlowSettings({ clearKey: false });
      await loadSiliconFlowSettings();
      const statusEl = RVC.byId("sfStatus");
      if (statusEl) {
        statusEl.textContent = `${statusEl.textContent} / 已保存`;
      }
    }).catch((err) => {
      RVC.setFormError("sfError", err.message);
    })
  );

  RVC.byId("btnSfClearKey").addEventListener("click", () =>
    RVC.runWithButton("btnSfClearKey", async () => {
      const ok = window.confirm("确认清空已保存的 API Key？");
      if (!ok) return;
      await saveSiliconFlowSettings({ clearKey: true });
      await loadSiliconFlowSettings();
    }).catch((err) => {
      RVC.setFormError("sfError", err.message);
    })
  );
}

async function initSettingsPage() {
  await RVC.loadHealth("healthBadge");
  wireSettingsActions();
  await loadSiliconFlowSettings();
}

initSettingsPage().catch((err) => {
  RVC.setFormError("sfError", err.message || String(err));
});

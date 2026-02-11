from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.backend.config import AppConfig


class SiliconFlowClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def available(self) -> bool:
        return bool(str(self.config.siliconflow_api_key or "").strip())

    def _compat_mode(self) -> str:
        mode = str(self.config.siliconflow_compat_mode or "").strip().lower()
        return "siliconflow" if mode == "siliconflow" else "strict_openai"

    def _request_max_tokens(self, requested: int) -> int:
        cap = max(64, int(getattr(self.config, "siliconflow_max_tokens", 4096)))
        return max(16, min(cap, int(requested)))

    def _endpoint(self) -> str:
        base = str(self.config.siliconflow_api_base or "").strip().rstrip("/")
        if not base:
            base = "https://api.siliconflow.cn/v1"
        return f"{base}/chat/completions"

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.8,
    ) -> str:
        api_key = str(self.config.siliconflow_api_key or "").strip()
        if not api_key:
            raise RuntimeError("SiliconFlow API key is missing (RVC_SILICONFLOW_API_KEY).")

        payload = {
            "model": str(self.config.siliconflow_model or "zai-org/GLM-4.5-Air"),
            "messages": messages,
            "stream": False,
            "max_tokens": self._request_max_tokens(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if self._compat_mode() != "strict_openai":
            payload["enable_thinking"] = bool(
                getattr(self.config, "siliconflow_enable_thinking", False)
            )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        timeout = max(2.0, float(self.config.siliconflow_timeout_s))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"SiliconFlow HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"SiliconFlow request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"SiliconFlow returned invalid JSON: {exc}") from exc
        return self._extract_text(data)

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("SiliconFlow response has no choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("SiliconFlow response has no message.")
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        return str(content or "").strip()

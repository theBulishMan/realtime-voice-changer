from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> Any:
        req_headers = dict(headers or {})
        req_body = body
        if json_payload is not None:
            req_body = json.dumps(json_payload).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=req_body,
            method=method,
            headers=req_headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                if not payload:
                    return {}
                return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = self._parse_http_error(exc)
            raise ApiError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"Request failed: {exc.reason}") from exc

    @staticmethod
    def _parse_http_error(exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            return str(exc)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip() or str(exc)
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        return raw.strip() or str(exc)

    def ping(self) -> bool:
        try:
            self.health()
            return True
        except ApiError:
            return False

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health")

    def audio_devices(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/audio/devices")

    def voices(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/v1/voices")
        if isinstance(payload, list):
            return payload
        raise ApiError("Unexpected response payload for voices")

    def realtime_state(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/realtime/state")

    def metrics_current(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/metrics/current")

    def update_realtime_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", "/api/v1/realtime/settings", json_payload=payload)

    def start_realtime(self, voice_id: str, language: str = "Auto") -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/realtime/start",
            json_payload={"voice_id": voice_id, "language": language},
        )

    def stop_realtime(self) -> dict[str, Any]:
        return self._request("POST", "/api/v1/realtime/stop")

    def create_design_voice(
        self,
        *,
        name: str,
        voice_prompt: str,
        preview_text: str,
        language: str = "Auto",
        save: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/voices/design",
            json_payload={
                "name": name,
                "voice_prompt": voice_prompt,
                "preview_text": preview_text,
                "language": language,
                "save": save,
            },
        )

    def create_clone_voice(
        self,
        *,
        name: str,
        ref_text: str,
        audio_path: str,
        language: str = "Auto",
    ) -> dict[str, Any]:
        file_path = Path(audio_path)
        if not file_path.exists():
            raise ApiError(f"Audio file does not exist: {audio_path}")
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        file_bytes = file_path.read_bytes()

        body, content_type_header = self._encode_multipart(
            fields={
                "name": name,
                "language": language,
                "ref_text": ref_text,
            },
            files={
                "audio_file": (file_path.name, content_type, file_bytes),
            },
        )
        return self._request(
            "POST",
            "/api/v1/voices/clone",
            headers={"Content-Type": content_type_header},
            body=body,
        )

    def preview_voice(self, voice_id: str, text: str, language: str = "Auto") -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/voices/{voice_id}/preview",
            json_payload={"text": text, "language": language},
        )

    def delete_voice(self, voice_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/voices/{voice_id}")

    @staticmethod
    def _encode_multipart(
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, str, bytes]],
    ) -> tuple[bytes, str]:
        boundary = f"----RVCBoundary{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for key, value in fields.items():
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8")
            )
        for key, (filename, content_type, file_bytes) in files.items():
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(
                (
                    f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8")
            )
            chunks.append(file_bytes)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(chunks)
        return body, f"multipart/form-data; boundary={boundary}"

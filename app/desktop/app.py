from __future__ import annotations

import base64
import binascii
from datetime import datetime
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from app.desktop.api_client import ApiClient, ApiError

winsound: Any
try:
    import winsound as _winsound
except Exception:
    winsound = None
else:
    winsound = _winsound


PROJECT_VIRTUAL_MIC_LABEL = "Realtime Voice Changer Virtual Mic"
DEFAULT_DESIGN_PROMPT_TEMPLATE = (
    "一位自然、真实、贴近人声的说话者，吐字清晰，语速中等偏稳，"
    "语气友好不过度夸张，情绪克制但有亲和力，句尾收音自然，避免电音感和机械感。"
)
DEFAULT_LONG_PREVIEW_TEXT = (
    "大家好，欢迎使用实时变声器。现在做一段通用试听：包含短句、长句、停顿和语气变化。"
    "如果你能清楚听到每个字并且觉得声音自然、不发闷、不刺耳，说明当前音色已经比较稳定。"
    "接下来我们继续读一段英文 mixed content for robustness check, including numbers like one two three."
)
UI_COLORS = {
    "bg": "#eef4ec",
    "surface": "#ffffff",
    "surface_alt": "#f7faf6",
    "ink": "#162316",
    "muted": "#4c5d4b",
    "line": "#c8d7c5",
    "accent": "#0b8f62",
    "accent_hover": "#097a54",
    "danger": "#d3662b",
    "danger_hover": "#bb5a25",
    "ok_bg": "#dff8ec",
    "ok_fg": "#126245",
    "warn_bg": "#fff1df",
    "warn_fg": "#9f5a1d",
}


class DesktopApp(tk.Tk):
    def __init__(self, base_url: str = "http://127.0.0.1:8787"):
        super().__init__()
        self.title("Realtime Voice Changer Desktop")
        self.geometry("1200x860")
        self.minsize(980, 760)

        self.api = ApiClient(base_url)
        self._backend_server: Any | None = None
        self._backend_thread: threading.Thread | None = None
        self._owns_backend_server = False
        self._closed = False
        self._poll_inflight = False
        self._poll_health_countdown = 0
        self._audio_preview_files: list[Path] = []

        self._all_device_labels: dict[str, int] = {}
        self._input_labels: list[str] = []
        self._output_labels: list[str] = []
        self._voice_labels: dict[str, str] = {}
        self._last_virtual_mic_id: int | None = None

        self.monitor_enabled_var = tk.BooleanVar(value=False)
        self.clone_audio_path_var = tk.StringVar(value="")
        self.health_text = tk.StringVar(value="Backend: checking...")
        self.gate_text = tk.StringVar(value="Latency Gate: waiting")
        self.device_notice_text = tk.StringVar(value="")
        self.metric_samples = tk.StringVar(value="Samples: 0")
        self.metric_fad = tk.StringVar(value="P95 FAD: 0ms")
        self.metric_e2e = tk.StringVar(value="P95 E2E: 0ms")
        self.metric_p50_fad = tk.StringVar(value="0.0ms")
        self.metric_p95_fad = tk.StringVar(value="0.0ms")
        self.metric_p50_e2e = tk.StringVar(value="0.0ms")
        self.metric_p95_e2e = tk.StringVar(value="0.0ms")

        self._init_theme()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._bootstrap)

    def _init_theme(self) -> None:
        self.configure(bg=UI_COLORS["bg"])
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("App.TFrame", background=UI_COLORS["bg"])
        style.configure(
            "Card.TLabelframe",
            background=UI_COLORS["surface"],
            bordercolor=UI_COLORS["line"],
            relief="solid",
            borderwidth=1,
            padding=10,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=UI_COLORS["surface"],
            foreground=UI_COLORS["ink"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Title.TLabel",
            background=UI_COLORS["surface"],
            foreground=UI_COLORS["ink"],
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "Subtle.TLabel",
            background=UI_COLORS["surface"],
            foreground=UI_COLORS["muted"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Field.TLabel",
            background=UI_COLORS["surface"],
            foreground=UI_COLORS["muted"],
            font=("Segoe UI", 9),
        )

        style.configure(
            "Primary.TButton",
            background=UI_COLORS["accent"],
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(12, 7),
        )
        style.map(
            "Primary.TButton",
            background=[
                ("disabled", "#c4d5ce"),
                ("pressed", UI_COLORS["accent_hover"]),
                ("active", UI_COLORS["accent_hover"]),
            ],
            foreground=[("disabled", "#f1f4f2"), ("active", "#ffffff"), ("pressed", "#ffffff")],
        )
        style.configure(
            "Danger.TButton",
            background=UI_COLORS["danger"],
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(12, 7),
        )
        style.map(
            "Danger.TButton",
            background=[
                ("disabled", "#d8c7bf"),
                ("pressed", UI_COLORS["danger_hover"]),
                ("active", UI_COLORS["danger_hover"]),
            ],
            foreground=[("disabled", "#f9f0ec"), ("active", "#ffffff"), ("pressed", "#ffffff")],
        )
        style.configure(
            "Ghost.TButton",
            background=UI_COLORS["surface_alt"],
            foreground=UI_COLORS["ink"],
            borderwidth=1,
            bordercolor=UI_COLORS["line"],
            padding=(12, 7),
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#ecf3eb"), ("pressed", "#e3ece2")],
            foreground=[("active", UI_COLORS["ink"]), ("pressed", UI_COLORS["ink"])],
        )

        style.configure(
            "App.TEntry",
            fieldbackground="#ffffff",
            foreground=UI_COLORS["ink"],
            bordercolor=UI_COLORS["line"],
            insertcolor=UI_COLORS["ink"],
        )
        style.configure(
            "App.TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            bordercolor=UI_COLORS["line"],
            arrowcolor=UI_COLORS["muted"],
            foreground=UI_COLORS["ink"],
        )
        style.map(
            "App.TCombobox",
            fieldbackground=[("readonly", "#ffffff")],
            selectbackground=[("readonly", UI_COLORS["accent"])],
            selectforeground=[("readonly", "#ffffff")],
        )
        style.configure(
            "App.TCheckbutton",
            background=UI_COLORS["surface"],
            foreground=UI_COLORS["muted"],
        )

        style.configure(
            "App.TNotebook",
            background=UI_COLORS["surface"],
            borderwidth=0,
            tabmargins=(0, 6, 0, 0),
        )
        style.configure(
            "App.TNotebook.Tab",
            background=UI_COLORS["surface_alt"],
            foreground=UI_COLORS["muted"],
            bordercolor=UI_COLORS["line"],
            padding=(12, 7),
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", UI_COLORS["surface"]), ("active", "#edf4ed")],
            foreground=[("selected", UI_COLORS["ink"]), ("active", UI_COLORS["ink"])],
        )

        style.configure(
            "BadgeNeutral.TLabel",
            background=UI_COLORS["surface_alt"],
            foreground=UI_COLORS["muted"],
            bordercolor=UI_COLORS["line"],
            relief="solid",
            borderwidth=1,
            padding=(10, 4),
            font=("Segoe UI", 9),
        )
        style.configure(
            "BadgeOk.TLabel",
            background=UI_COLORS["ok_bg"],
            foreground=UI_COLORS["ok_fg"],
            bordercolor="#7fc8a8",
            relief="solid",
            borderwidth=1,
            padding=(10, 4),
            font=("Segoe UI", 9),
        )
        style.configure(
            "BadgeWarn.TLabel",
            background=UI_COLORS["warn_bg"],
            foreground=UI_COLORS["warn_fg"],
            bordercolor="#efbf84",
            relief="solid",
            borderwidth=1,
            padding=(10, 4),
            font=("Segoe UI", 9),
        )
        style.configure(
            "MetricLabel.TLabel",
            background=UI_COLORS["surface_alt"],
            foreground=UI_COLORS["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "MetricValue.TLabel",
            background=UI_COLORS["surface_alt"],
            foreground=UI_COLORS["ink"],
            font=("Segoe UI Semibold", 16),
        )

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(4, weight=1)

        hero = tk.Frame(
            root,
            bg=UI_COLORS["surface"],
            highlightbackground=UI_COLORS["line"],
            highlightthickness=1,
            bd=0,
            padx=16,
            pady=14,
        )
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        chip = tk.Label(
            hero,
            text="LOCAL / QWEN3-TTS / DESKTOP",
            bg=UI_COLORS["surface_alt"],
            fg=UI_COLORS["muted"],
            font=("Consolas", 9),
            padx=8,
            pady=2,
        )
        chip.pack(anchor="w")

        hero_title_row = tk.Frame(hero, bg=UI_COLORS["surface"])
        hero_title_row.pack(fill="x", pady=(8, 4))
        tk.Label(
            hero_title_row,
            text="Realtime Voice Changer",
            bg=UI_COLORS["surface"],
            fg=UI_COLORS["ink"],
            font=("Segoe UI Semibold", 20),
        ).pack(side="left")
        ttk.Button(hero_title_row, text="Refresh", command=self._refresh_all, style="Ghost.TButton").pack(
            side="right"
        )
        tk.Label(
            hero,
            text="Desktop runtime for local voice design, cloning, preview, and realtime conversion.",
            bg=UI_COLORS["surface"],
            fg=UI_COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        status_strip = ttk.Frame(root, style="App.TFrame")
        status_strip.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.health_badge = ttk.Label(
            status_strip, textvariable=self.health_text, style="BadgeNeutral.TLabel"
        )
        self.health_badge.pack(side="left")
        self.gate_badge = ttk.Label(
            status_strip, textvariable=self.gate_text, style="BadgeNeutral.TLabel"
        )
        self.gate_badge.pack(side="left", padx=(8, 0))
        ttk.Label(status_strip, textvariable=self.metric_samples, style="BadgeNeutral.TLabel").pack(
            side="left", padx=(8, 0)
        )

        realtime = ttk.LabelFrame(root, text="Realtime Control", style="Card.TLabelframe")
        realtime.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        for idx in range(3):
            realtime.columnconfigure(idx, weight=1)

        ttk.Label(realtime, text="Input Device", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        ttk.Label(realtime, text="Virtual Mic Output", style="Field.TLabel").grid(
            row=0, column=1, sticky="w", pady=(0, 2)
        )
        ttk.Label(realtime, text="Monitor Output", style="Field.TLabel").grid(
            row=0, column=2, sticky="w", pady=(0, 2)
        )
        self.input_combo = ttk.Combobox(realtime, state="readonly", style="App.TCombobox")
        self.input_combo.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.virtual_combo = ttk.Combobox(realtime, state="readonly", style="App.TCombobox")
        self.virtual_combo.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.monitor_combo = ttk.Combobox(realtime, state="readonly", style="App.TCombobox")
        self.monitor_combo.grid(row=1, column=2, sticky="ew", pady=(0, 8))

        self.device_notice = ttk.Label(
            realtime, textvariable=self.device_notice_text, style="BadgeNeutral.TLabel"
        )
        self.device_notice.grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(realtime, text="Language", style="Field.TLabel").grid(
            row=3, column=0, sticky="w", pady=(0, 2)
        )
        ttk.Label(realtime, text="VAD Silence (ms)", style="Field.TLabel").grid(
            row=3, column=1, sticky="w", pady=(0, 2)
        )
        ttk.Label(realtime, text="Max Segment (ms)", style="Field.TLabel").grid(
            row=3, column=2, sticky="w", pady=(0, 2)
        )
        self.language_combo = ttk.Combobox(
            realtime,
            state="readonly",
            values=["Auto", "Chinese", "English"],
            style="App.TCombobox",
        )
        self.language_combo.set("Auto")
        self.language_combo.grid(row=4, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.vad_entry = ttk.Entry(realtime, style="App.TEntry")
        self.vad_entry.insert(0, "240")
        self.vad_entry.grid(row=4, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.max_segment_entry = ttk.Entry(realtime, style="App.TEntry")
        self.max_segment_entry.insert(0, "12000")
        self.max_segment_entry.grid(row=4, column=2, sticky="ew", pady=(0, 8))

        action_row = ttk.Frame(realtime, style="App.TFrame")
        action_row.grid(row=5, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(
            action_row,
            text="Enable local monitor playback",
            variable=self.monitor_enabled_var,
            style="App.TCheckbutton",
        ).pack(side="left")
        ttk.Button(action_row, text="Apply Settings", command=self._apply_settings, style="Ghost.TButton").pack(
            side="right"
        )
        ttk.Button(action_row, text="Stop", command=self._stop_realtime, style="Danger.TButton").pack(
            side="right", padx=(0, 8)
        )
        ttk.Button(action_row, text="Start", command=self._start_realtime, style="Primary.TButton").pack(
            side="right", padx=(0, 8)
        )

        voice = ttk.LabelFrame(root, text="Voice", style="Card.TLabelframe")
        voice.grid(row=2, column=1, sticky="nsew", pady=(0, 8))
        voice.columnconfigure(0, weight=1)
        ttk.Label(voice, text="Current Voice", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.voice_combo = ttk.Combobox(voice, state="readonly", style="App.TCombobox")
        self.voice_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))

        voice_actions = ttk.Frame(voice, style="App.TFrame")
        voice_actions.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(voice_actions, text="Refresh Voices", command=self._refresh_voices, style="Ghost.TButton").pack(
            side="left"
        )
        ttk.Button(voice_actions, text="Delete Voice", command=self._delete_voice, style="Danger.TButton").pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(voice, text="Preview Text", style="Field.TLabel").grid(row=3, column=0, sticky="w")
        self.preview_text = tk.Text(
            voice,
            height=4,
            wrap="word",
            bg=UI_COLORS["surface_alt"],
            fg=UI_COLORS["ink"],
            insertbackground=UI_COLORS["ink"],
            highlightbackground=UI_COLORS["line"],
            highlightthickness=1,
            relief="flat",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
        )
        self.preview_text.insert("1.0", DEFAULT_LONG_PREVIEW_TEXT)
        self.preview_text.grid(row=4, column=0, sticky="ew", pady=(2, 8))
        ttk.Button(voice, text="Preview Voice", command=self._preview_voice, style="Primary.TButton").grid(
            row=5, column=0, sticky="e"
        )

        manage = ttk.LabelFrame(root, text="Create Voice", style="Card.TLabelframe")
        manage.grid(row=3, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        manage.columnconfigure(0, weight=1)
        tabs = ttk.Notebook(manage, style="App.TNotebook")
        tabs.grid(row=0, column=0, sticky="ew")

        design_tab = ttk.Frame(tabs, style="App.TFrame", padding=8)
        clone_tab = ttk.Frame(tabs, style="App.TFrame", padding=8)
        tabs.add(design_tab, text="Design")
        tabs.add(clone_tab, text="Clone")

        self.design_name = ttk.Entry(design_tab, style="App.TEntry")
        self.design_prompt = tk.Text(
            design_tab,
            height=4,
            wrap="word",
            bg=UI_COLORS["surface_alt"],
            fg=UI_COLORS["ink"],
            insertbackground=UI_COLORS["ink"],
            highlightbackground=UI_COLORS["line"],
            highlightthickness=1,
            relief="flat",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
        )
        self.design_preview_text = tk.Text(
            design_tab,
            height=3,
            wrap="word",
            bg=UI_COLORS["surface_alt"],
            fg=UI_COLORS["ink"],
            insertbackground=UI_COLORS["ink"],
            highlightbackground=UI_COLORS["line"],
            highlightthickness=1,
            relief="flat",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
        )
        self.design_language = ttk.Combobox(
            design_tab, state="readonly", values=["Auto", "Chinese", "English"], style="App.TCombobox"
        )
        self.design_language.set("Auto")

        ttk.Label(design_tab, text="Name", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.design_name.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(design_tab, text="Voice Prompt", style="Field.TLabel").grid(row=2, column=0, sticky="w")
        self.design_prompt.grid(row=3, column=0, sticky="ew", pady=(2, 6))
        self.design_prompt.insert("1.0", DEFAULT_DESIGN_PROMPT_TEMPLATE)
        ttk.Label(design_tab, text="Preview Text", style="Field.TLabel").grid(row=4, column=0, sticky="w")
        self.design_preview_text.grid(row=5, column=0, sticky="ew", pady=(2, 6))
        self.design_preview_text.insert("1.0", DEFAULT_LONG_PREVIEW_TEXT)
        ttk.Label(design_tab, text="Language", style="Field.TLabel").grid(row=6, column=0, sticky="w")
        self.design_language.grid(row=7, column=0, sticky="ew")
        d_actions = ttk.Frame(design_tab, style="App.TFrame")
        d_actions.grid(row=8, column=0, sticky="e", pady=(8, 0))
        ttk.Button(d_actions, text="Preview Only", command=self._design_preview, style="Ghost.TButton").pack(
            side="left"
        )
        ttk.Button(d_actions, text="Create & Save", command=self._design_save, style="Primary.TButton").pack(
            side="left", padx=(8, 0)
        )
        design_tab.columnconfigure(0, weight=1)

        self.clone_name = ttk.Entry(clone_tab, style="App.TEntry")
        self.clone_ref_text = tk.Text(
            clone_tab,
            height=3,
            wrap="word",
            bg=UI_COLORS["surface_alt"],
            fg=UI_COLORS["ink"],
            insertbackground=UI_COLORS["ink"],
            highlightbackground=UI_COLORS["line"],
            highlightthickness=1,
            relief="flat",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
        )
        self.clone_language = ttk.Combobox(
            clone_tab, state="readonly", values=["Auto", "Chinese", "English"], style="App.TCombobox"
        )
        self.clone_language.set("Auto")
        self.clone_file_entry = ttk.Entry(clone_tab, textvariable=self.clone_audio_path_var, style="App.TEntry")
        ttk.Label(clone_tab, text="Name", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.clone_name.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(clone_tab, text="Reference Text", style="Field.TLabel").grid(row=2, column=0, sticky="w")
        self.clone_ref_text.grid(row=3, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(clone_tab, text="Audio File", style="Field.TLabel").grid(row=4, column=0, sticky="w")
        file_row = ttk.Frame(clone_tab, style="App.TFrame")
        file_row.grid(row=5, column=0, sticky="ew", pady=(2, 6))
        file_row.columnconfigure(0, weight=1)
        self.clone_file_entry.grid(in_=file_row, row=0, column=0, sticky="ew")
        ttk.Button(file_row, text="Browse", command=self._browse_clone_audio, style="Ghost.TButton").grid(
            row=0, column=1, padx=(8, 0)
        )
        ttk.Label(clone_tab, text="Language", style="Field.TLabel").grid(row=6, column=0, sticky="w")
        self.clone_language.grid(row=7, column=0, sticky="ew")
        ttk.Button(clone_tab, text="Create Clone", command=self._clone_create, style="Primary.TButton").grid(
            row=8, column=0, sticky="e", pady=(8, 0)
        )
        clone_tab.columnconfigure(0, weight=1)

        metrics = ttk.LabelFrame(root, text="Metrics", style="Card.TLabelframe")
        metrics.grid(row=3, column=1, sticky="nsew", pady=(0, 8))
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        ttk.Label(metrics, textvariable=self.metric_fad, style="Subtle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(metrics, textvariable=self.metric_e2e, style="Subtle.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        metric_cards = ttk.Frame(metrics, style="App.TFrame")
        metric_cards.grid(row=2, column=0, columnspan=2, sticky="ew")
        metric_cards.columnconfigure(0, weight=1)
        metric_cards.columnconfigure(1, weight=1)

        def add_metric_tile(row: int, col: int, title: str, value_var: tk.StringVar) -> None:
            tile = tk.Frame(
                metric_cards,
                bg=UI_COLORS["surface_alt"],
                highlightbackground=UI_COLORS["line"],
                highlightthickness=1,
                padx=10,
                pady=8,
            )
            tile.grid(
                row=row,
                column=col,
                sticky="nsew",
                padx=(0 if col == 0 else 4, 4 if col == 0 else 0),
                pady=(0 if row == 0 else 4, 0),
            )
            tk.Label(
                tile,
                text=title,
                bg=UI_COLORS["surface_alt"],
                fg=UI_COLORS["muted"],
                font=("Segoe UI", 9),
            ).pack(anchor="w")
            ttk.Label(tile, textvariable=value_var, style="MetricValue.TLabel").pack(anchor="w")

        add_metric_tile(0, 0, "P50 FAD", self.metric_p50_fad)
        add_metric_tile(0, 1, "P95 FAD", self.metric_p95_fad)
        add_metric_tile(1, 0, "P50 E2E", self.metric_p50_e2e)
        add_metric_tile(1, 1, "P95 E2E", self.metric_p95_e2e)

        logs = ttk.LabelFrame(root, text="Events", style="Card.TLabelframe")
        logs.grid(row=4, column=0, columnspan=2, sticky="nsew")
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            logs,
            height=14,
            wrap="word",
            bg=UI_COLORS["surface_alt"],
            fg=UI_COLORS["ink"],
            insertbackground=UI_COLORS["ink"],
            highlightbackground=UI_COLORS["line"],
            highlightthickness=1,
            relief="flat",
            font=("Consolas", 9),
            padx=10,
            pady=10,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")
    def _bootstrap(self) -> None:
        if self.api.ping():
            self._set_health("Connected to backend (existing process)")
            self._log("Connected to existing backend process.")
            self._refresh_all()
            self._schedule_poll()
            return
        self._log("No running backend found, starting embedded backend server.")
        self._start_embedded_backend()
        self._wait_for_backend_ready(0)

    def _start_embedded_backend(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            "app.backend.main:app",
            host="127.0.0.1",
            port=8787,
            reload=False,
            log_level="warning",
        )
        self._backend_server = uvicorn.Server(config)
        self._owns_backend_server = True
        self._backend_thread = threading.Thread(target=self._backend_server.run, daemon=True)
        self._backend_thread.start()

    def _wait_for_backend_ready(self, attempt: int) -> None:
        if self.api.ping():
            self._set_health("Embedded backend ready")
            self._log("Embedded backend is ready.")
            self._refresh_all()
            self._schedule_poll()
            return
        if attempt >= 120:
            self._set_health("Backend startup timeout")
            messagebox.showerror(
                "Backend Startup Failed",
                "Failed to start embedded backend. Check logs and environment.",
            )
            return
        self.after(500, lambda: self._wait_for_backend_ready(attempt + 1))

    def _run_task(
        self,
        *,
        name: str,
        action: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        silent: bool = False,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        def worker() -> None:
            try:
                result = action()
            except Exception as exc:
                err = exc

                def notify_error() -> None:
                    self._on_task_error(name, err, silent=silent)

                self.after(0, notify_error)
            else:
                if on_success is not None:
                    self.after(0, lambda: on_success(result))
            finally:
                if on_done is not None:
                    self.after(0, on_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_task_error(self, name: str, exc: Exception, *, silent: bool) -> None:
        if not silent:
            self._log(f"{name} failed: {exc}")
            messagebox.showerror("Operation Failed", f"{name} failed:\n{exc}")

    def _refresh_all(self) -> None:
        def action() -> dict[str, Any]:
            return {
                "health": self.api.health(),
                "devices": self.api.audio_devices(),
                "voices": self.api.voices(),
                "state": self.api.realtime_state(),
                "metrics": self.api.metrics_current(),
            }

        self._run_task(name="refresh", action=action, on_success=self._apply_snapshot)

    def _apply_snapshot(self, payload: dict[str, Any]) -> None:
        self._apply_health(payload["health"])
        self._apply_devices(payload["devices"])
        self._apply_voices(payload["voices"])
        self._apply_realtime_state(payload["state"])
        self._apply_metrics(payload["metrics"])
        self._log("Snapshot refreshed.")

    def _apply_health(self, health: dict[str, Any]) -> None:
        status = str(health.get("status", "unknown"))
        fake_mode = bool(health.get("fake_mode", False))
        mode = "fake" if fake_mode else "real"
        gpu_required = bool(health.get("gpu_required", False))
        gpu_available = bool(health.get("gpu_available", False))
        gpu_device = str(health.get("gpu_device") or "")
        if gpu_required:
            gpu_suffix = f", gpu={'ok' if gpu_available else 'missing'}"
            if gpu_available and gpu_device:
                gpu_suffix = f"{gpu_suffix} ({gpu_device})"
        else:
            gpu_suffix = ""
        if status == "ok":
            self.health_badge.configure(style="BadgeOk.TLabel")
            self._set_health(f"Backend ready ({mode} mode{gpu_suffix})")
        elif status == "degraded":
            self.health_badge.configure(style="BadgeWarn.TLabel")
            self._set_health(f"Backend degraded ({mode} mode{gpu_suffix})")
        else:
            self.health_badge.configure(style="BadgeNeutral.TLabel")
            self._set_health(f"Backend status: {status} ({mode} mode{gpu_suffix})")

    def _apply_devices(self, payload: dict[str, Any]) -> None:
        devices = payload.get("devices", [])
        default_input = payload.get("default_input_id")
        default_output = payload.get("default_output_id")
        virtual_mic_id = payload.get("virtual_mic_output_id")

        self._all_device_labels.clear()
        self._input_labels = []
        self._output_labels = []
        self._last_virtual_mic_id = int(virtual_mic_id) if isinstance(virtual_mic_id, int) else None

        for dev in devices:
            dev_id = int(dev["id"])
            label = str(dev.get("name", f"Device {dev_id}"))
            if self._last_virtual_mic_id is not None and dev_id == self._last_virtual_mic_id:
                label = f"{PROJECT_VIRTUAL_MIC_LABEL} ({label})"
            if bool(dev.get("is_default_input", False)):
                label = f"{label} [Default]"
            if bool(dev.get("is_default_output", False)):
                label = f"{label} [Default]"
            unique_label = self._register_unique_device_label(label, dev_id)

            if int(dev.get("max_input_channels", 0)) > 0:
                self._input_labels.append(unique_label)
            if int(dev.get("max_output_channels", 0)) > 0:
                self._output_labels.append(unique_label)

        self.input_combo["values"] = self._input_labels
        self.virtual_combo["values"] = self._output_labels
        self.monitor_combo["values"] = self._output_labels

        self._set_combo_by_id(self.input_combo, default_input)
        self._set_combo_by_id(self.monitor_combo, default_output)
        self._set_combo_by_id(self.virtual_combo, virtual_mic_id)

        if self._last_virtual_mic_id is None:
            self.device_notice.configure(style="BadgeWarn.TLabel")
            self.device_notice_text.set("Virtual mic not detected. Install VB-CABLE and restart app.")
            self._log("Virtual mic not detected yet (VB-CABLE missing).")
        else:
            self.device_notice.configure(style="BadgeOk.TLabel")
            self.device_notice_text.set("Virtual mic detected and ready.")

    def _apply_voices(self, voices: list[dict[str, Any]]) -> None:
        self._voice_labels.clear()
        labels: list[str] = []
        for voice in voices:
            voice_id = str(voice.get("id", ""))
            name = str(voice.get("name", voice_id))
            mode = str(voice.get("mode", "unknown"))
            label = self._register_unique_voice_label(f"{name} ({mode})", voice_id)
            labels.append(label)

        self.voice_combo["values"] = labels
        if labels and not self.voice_combo.get():
            self.voice_combo.set(labels[0])

    def _apply_realtime_state(self, state: dict[str, Any]) -> None:
        settings = state.get("settings", {})
        self.monitor_enabled_var.set(bool(settings.get("monitor_enabled", False)))
        self.vad_entry.delete(0, "end")
        self.vad_entry.insert(0, str(settings.get("vad_silence_ms", 240)))
        self.max_segment_entry.delete(0, "end")
        self.max_segment_entry.insert(0, str(settings.get("max_segment_ms", 12000)))
        self._set_combo_by_id(self.input_combo, settings.get("input_device_id"))
        self._set_combo_by_id(self.virtual_combo, settings.get("virtual_mic_device_id"))
        self._set_combo_by_id(self.monitor_combo, settings.get("monitor_device_id"))
        voice_id = state.get("voice_id")
        if isinstance(voice_id, str):
            self._set_voice_by_id(voice_id)

    def _apply_metrics(self, metrics: dict[str, Any]) -> None:
        p50_fad = float(metrics.get("p50_fad_ms", 0.0))
        p95_fad = float(metrics.get("p95_fad_ms", 0.0))
        p50_e2e = float(metrics.get("p50_e2e_ms", 0.0))
        p95_e2e = float(metrics.get("p95_e2e_ms", 0.0))
        samples = int(metrics.get("sample_count", 0))
        self.metric_samples.set(f"Samples: {samples}")
        self.metric_fad.set(f"P95 FAD: {p95_fad:.1f}ms")
        self.metric_e2e.set(f"P95 E2E: {p95_e2e:.1f}ms")
        self.metric_p50_fad.set(f"{p50_fad:.1f}ms")
        self.metric_p95_fad.set(f"{p95_fad:.1f}ms")
        self.metric_p50_e2e.set(f"{p50_e2e:.1f}ms")
        self.metric_p95_e2e.set(f"{p95_e2e:.1f}ms")
        fad_ok = p95_fad <= 250.0
        e2e_ok = p95_e2e <= 800.0
        if samples <= 0:
            self.gate_text.set("Latency Gate: waiting for samples")
            self.gate_badge.configure(style="BadgeNeutral.TLabel")
        else:
            self.gate_text.set(
                f"Latency Gate: FAD {'PASS' if fad_ok else 'FAIL'} / "
                f"E2E {'PASS' if e2e_ok else 'FAIL'}"
            )
            self.gate_badge.configure(style="BadgeOk.TLabel" if fad_ok and e2e_ok else "BadgeWarn.TLabel")

    def _refresh_voices(self) -> None:
        self._run_task(name="refresh voices", action=self.api.voices, on_success=self._apply_voices)

    def _apply_settings(self) -> None:
        try:
            payload = {
                "input_device_id": self._selected_device_id(self.input_combo),
                "monitor_device_id": self._selected_device_id(self.monitor_combo),
                "virtual_mic_device_id": self._selected_device_id(self.virtual_combo),
                "monitor_enabled": bool(self.monitor_enabled_var.get()),
                "vad_silence_ms": int(self.vad_entry.get().strip()),
                "max_segment_ms": int(self.max_segment_entry.get().strip()),
            }
        except ValueError:
            messagebox.showerror("Invalid Input", "VAD and Max Segment must be integers.")
            return
        self._run_task(
            name="apply settings",
            action=lambda: self.api.update_realtime_settings(payload),
            on_success=lambda _: self._log("Realtime settings updated."),
        )

    def _start_realtime(self) -> None:
        voice_id = self._selected_voice_id()
        if not voice_id:
            messagebox.showwarning("Missing Voice", "Select a voice before starting realtime.")
            return
        language = self.language_combo.get().strip() or "Auto"
        self._run_task(
            name="start realtime",
            action=lambda: self.api.start_realtime(voice_id=voice_id, language=language),
            on_success=lambda _: self._log("Realtime started."),
        )

    def _stop_realtime(self) -> None:
        self._run_task(
            name="stop realtime",
            action=self.api.stop_realtime,
            on_success=lambda _: self._log("Realtime stopped."),
        )

    def _preview_voice(self) -> None:
        voice_id = self._selected_voice_id()
        text = self.preview_text.get("1.0", "end").strip()
        if not voice_id:
            messagebox.showwarning("Missing Voice", "Select a voice for preview.")
            return
        if not text:
            messagebox.showwarning("Missing Text", "Preview text cannot be empty.")
            return

        def on_success(payload: dict[str, Any]) -> None:
            audio_b64 = payload.get("audio_b64")
            if not isinstance(audio_b64, str):
                raise ApiError("Preview response does not contain audio payload.")
            self._play_preview_audio(audio_b64)
            self._log(f"Preview generated for voice={voice_id}.")

        self._run_task(
            name="preview voice",
            action=lambda: self.api.preview_voice(voice_id=voice_id, text=text),
            on_success=on_success,
        )

    def _delete_voice(self) -> None:
        voice_id = self._selected_voice_id()
        if not voice_id:
            messagebox.showwarning("Missing Voice", "Select a voice first.")
            return
        if not messagebox.askyesno("Delete Voice", f"Delete voice {voice_id}?"):
            return

        def on_success(_: Any) -> None:
            self._log(f"Voice deleted: {voice_id}")
            self._refresh_voices()

        self._run_task(
            name="delete voice",
            action=lambda: self.api.delete_voice(voice_id),
            on_success=on_success,
        )

    def _design_preview(self) -> None:
        self._create_design_voice(save=False)

    def _design_save(self) -> None:
        self._create_design_voice(save=True)

    def _create_design_voice(self, *, save: bool) -> None:
        name = self.design_name.get().strip()
        prompt = self.design_prompt.get("1.0", "end").strip()
        preview_text = self.design_preview_text.get("1.0", "end").strip()
        language = self.design_language.get().strip() or "Auto"
        if not name or not prompt or not preview_text:
            messagebox.showwarning("Missing Fields", "Design voice fields cannot be empty.")
            return

        def on_success(payload: dict[str, Any]) -> None:
            audio_b64 = payload.get("preview_audio_b64")
            if isinstance(audio_b64, str):
                self._play_preview_audio(audio_b64)
            if save:
                voice = payload.get("voice", {})
                saved_id = voice.get("id")
                self._log(f"Design voice saved: {voice.get('name', saved_id)}")
                self._refresh_voices()
                if isinstance(saved_id, str):
                    self.after(350, lambda: self._set_voice_by_id(saved_id))
            else:
                self._log("Design preview generated (not saved).")

        self._run_task(
            name="create design voice",
            action=lambda: self.api.create_design_voice(
                name=name,
                voice_prompt=prompt,
                preview_text=preview_text,
                language=language,
                save=save,
            ),
            on_success=on_success,
        )

    def _browse_clone_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select reference audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.m4a"),
                ("WAV", "*.wav"),
                ("MP3", "*.mp3"),
                ("M4A", "*.m4a"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.clone_audio_path_var.set(path)

    def _clone_create(self) -> None:
        name = self.clone_name.get().strip()
        ref_text = self.clone_ref_text.get("1.0", "end").strip()
        audio_path = self.clone_audio_path_var.get().strip()
        language = self.clone_language.get().strip() or "Auto"
        if not name or not ref_text or not audio_path:
            messagebox.showwarning("Missing Fields", "Clone fields cannot be empty.")
            return

        def on_success(payload: dict[str, Any]) -> None:
            audio_b64 = payload.get("preview_audio_b64")
            if isinstance(audio_b64, str):
                self._play_preview_audio(audio_b64)
            voice = payload.get("voice", {})
            saved_id = voice.get("id")
            self._log(f"Clone voice saved: {voice.get('name', saved_id)}")
            self._refresh_voices()
            if isinstance(saved_id, str):
                self.after(350, lambda: self._set_voice_by_id(saved_id))

        self._run_task(
            name="create clone voice",
            action=lambda: self.api.create_clone_voice(
                name=name,
                ref_text=ref_text,
                audio_path=audio_path,
                language=language,
            ),
            on_success=on_success,
        )

    def _play_preview_audio(self, audio_b64: str) -> None:
        try:
            wav = base64.b64decode(audio_b64.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ApiError(f"Invalid audio payload: {exc}") from exc

        if winsound is None:
            self._log("winsound is unavailable on this platform. Preview playback skipped.")
            return

        temp_dir = Path.cwd() / "tmp" / "preview_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / f"preview_{threading.get_ident()}_{len(self._audio_preview_files)}.wav"
        file_path.write_bytes(wav)
        self._audio_preview_files.append(file_path)
        winsound.PlaySound(str(file_path), winsound.SND_FILENAME | winsound.SND_ASYNC)

    def _schedule_poll(self) -> None:
        self.after(1200, self._poll_tick)

    def _poll_tick(self) -> None:
        if self._closed:
            return
        if self._poll_inflight:
            self.after(1200, self._poll_tick)
            return
        self._poll_inflight = True

        def action() -> dict[str, Any]:
            payload = {
                "state": self.api.realtime_state(),
                "metrics": self.api.metrics_current(),
            }
            if self._poll_health_countdown <= 0:
                payload["health"] = self.api.health()
            return payload

        def on_success(payload: dict[str, Any]) -> None:
            self._apply_realtime_state(payload["state"])
            self._apply_metrics(payload["metrics"])
            if "health" in payload:
                self._apply_health(payload["health"])
                self._poll_health_countdown = 10
            else:
                self._poll_health_countdown -= 1

        def on_done() -> None:
            self._poll_inflight = False
            if not self._closed:
                self.after(1200, self._poll_tick)

        self._run_task(
            name="poll runtime",
            action=action,
            on_success=on_success,
            silent=True,
            on_done=on_done,
        )

    def _selected_device_id(self, combo: ttk.Combobox) -> int | None:
        label = combo.get().strip()
        if not label:
            return None
        return self._all_device_labels.get(label)

    def _selected_voice_id(self) -> str | None:
        label = self.voice_combo.get().strip()
        if not label:
            return None
        return self._voice_labels.get(label)

    def _set_combo_by_id(self, combo: ttk.Combobox, device_id: Any) -> None:
        if not isinstance(device_id, int):
            return
        for label, mapped_id in self._all_device_labels.items():
            if mapped_id == device_id:
                combo.set(label)
                return

    def _set_voice_by_id(self, voice_id: str) -> None:
        for label, mapped_id in self._voice_labels.items():
            if mapped_id == voice_id:
                self.voice_combo.set(label)
                return

    def _register_unique_device_label(self, base_label: str, device_id: int) -> str:
        if base_label not in self._all_device_labels:
            self._all_device_labels[base_label] = device_id
            return base_label
        suffix = 2
        while True:
            candidate = f"{base_label} ({suffix})"
            if candidate not in self._all_device_labels:
                self._all_device_labels[candidate] = device_id
                return candidate
            suffix += 1

    def _register_unique_voice_label(self, base_label: str, voice_id: str) -> str:
        if base_label not in self._voice_labels:
            self._voice_labels[base_label] = voice_id
            return base_label
        suffix = 2
        while True:
            candidate = f"{base_label} ({suffix})"
            if candidate not in self._voice_labels:
                self._voice_labels[candidate] = voice_id
                return candidate
            suffix += 1

    def _set_health(self, text: str) -> None:
        self.health_text.set(text)

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("1.0", f"[{timestamp}] {message}\n")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        self._closed = True
        if self._backend_server is not None and self._owns_backend_server:
            self._backend_server.should_exit = True
        for file_path in self._audio_preview_files:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
        self.destroy()


def launch_desktop_app(base_url: str = "http://127.0.0.1:8787") -> None:
    app = DesktopApp(base_url=base_url)
    app.mainloop()


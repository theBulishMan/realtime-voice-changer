from __future__ import annotations

from PyInstaller.utils.hooks import copy_metadata

# webrtcvad-wheels publishes module "webrtcvad" but distribution metadata
# lives under "webrtcvad-wheels". This avoids hook failures in contrib hooks.
datas = copy_metadata("webrtcvad-wheels")
hiddenimports = ["_webrtcvad"]

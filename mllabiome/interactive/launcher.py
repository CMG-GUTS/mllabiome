from __future__ import annotations

import shutil
import subprocess
import threading
import webbrowser
from pathlib import Path

from .api import create_server
from .runs import RunStore


FRONTEND = Path("interactive/frontend")


def frontend_directory(config: Path) -> Path:
    for root in (*config.expanduser().resolve().parents, Path(__file__).resolve().parents[2]):
        candidate = root / FRONTEND
        if (candidate / "package-lock.json").is_file() and (candidate / "app" / "inference" / "page.tsx").is_file():
            return candidate
    raise FileNotFoundError("Cannot find interactive/frontend in the mllabiome source checkout.")


def frontend_assets(config: Path) -> Path:
    frontend = frontend_directory(config)
    output = frontend / "out"
    if (output / "index.html").is_file() and (output / "inference" / "index.html").is_file():
        return output
    if not shutil.which("npm") or not shutil.which("node"):
        raise RuntimeError("Interactive inference requires Node.js 20+ and npm on first launch to build the original TypeScript frontend.")
    try:
        if not (frontend / "node_modules" / ".bin" / "next").is_file():
            subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True)
        subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("The original mllabiome-ii frontend could not be built. Check the npm output above.") from exc
    if not (output / "inference" / "index.html").is_file():
        raise RuntimeError("The interactive frontend build did not produce the inference page.")
    return output


def launch(root: Path, config: Path, port: int = 8765, open_browser: bool = True) -> None:
    RunStore(root)
    assets = frontend_assets(config)
    with create_server(root, port, static=assets) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        print(f"Interactive inference · {url}")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

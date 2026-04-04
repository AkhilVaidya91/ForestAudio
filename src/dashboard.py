from __future__ import annotations

import webbrowser

from src.config import AppConfig


def main() -> None:
    config = AppConfig()
    url = f"http://{config.cloud_host}:{config.cloud_port}/"
    print(f"Open the ForestAudio dashboard at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    main()

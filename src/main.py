from __future__ import annotations

import argparse
import threading
import time

from werkzeug.serving import make_server

from .cloud import create_app
from .config import AppConfig
from .simulator import run_simulator
from .storage import LocalStore
from .worker import run_worker


class FlaskServerThread(threading.Thread):
    def __init__(self, app, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.server = make_server(host, port, app)
        self.context = app.app_context()

    def run(self) -> None:
        self.context.push()
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


def run_api(config: AppConfig, store: LocalStore) -> None:
    app = create_app(config, store)
    app.run(host=config.cloud_host, port=config.cloud_port, debug=False, use_reloader=False)


def run_all(config: AppConfig, store: LocalStore) -> None:
    app = create_app(config, store)
    server = FlaskServerThread(app=app, host=config.cloud_host, port=config.cloud_port)
    server.start()
    worker_thread = threading.Thread(target=run_worker, args=(config, store), daemon=True)
    simulator_thread = threading.Thread(target=run_simulator, args=(config, store), daemon=True)
    worker_thread.start()
    simulator_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ForestAudio local acoustic monitoring stack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("api")
    subparsers.add_parser("worker")
    subparsers.add_parser("simulate")
    subparsers.add_parser("run-all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig()
    store = LocalStore(config.db_path)
    if args.command == "api":
        run_api(config, store)
    elif args.command == "worker":
        run_worker(config, store)
    elif args.command == "simulate":
        run_simulator(config, store)
    elif args.command == "run-all":
        run_all(config, store)


if __name__ == "__main__":
    main()

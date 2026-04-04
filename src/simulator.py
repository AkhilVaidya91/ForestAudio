from __future__ import annotations

import random
import time
from dataclasses import dataclass

import requests

from .config import AppConfig
from .edge import EdgeModelEngine
from .models import NodeState
from .sampler import EnvironmentSampler
from .storage import LocalStore


@dataclass(slots=True)
class SensorNode:
    state: NodeState


class EdgeNetworkSimulator:
    def __init__(self, config: AppConfig, store: LocalStore) -> None:
        self.config = config
        self.store = store
        self.rng = random.Random(42)
        self.sampler = EnvironmentSampler(config, self.rng)
        self.edge_engine = EdgeModelEngine(config)
        self.nodes = self._build_nodes()

    def _build_nodes(self) -> list[SensorNode]:
        nodes: list[SensorNode] = []
        for index in range(self.config.max_nodes):
            lat = self.config.center_lat + self.rng.uniform(-self.config.map_span, self.config.map_span)
            lon = self.config.center_lon + self.rng.uniform(-self.config.map_span, self.config.map_span)
            node_id = f"node-{index + 1:02d}"
            state = NodeState(node_id=node_id, lat=lat, lon=lon)
            nodes.append(SensorNode(state=state))
            self.store.upsert_node(node_id=node_id, lat=lat, lon=lon, status="active")
        return nodes

    def _cloud_url(self, path: str) -> str:
        return f"http://{self.config.cloud_host}:{self.config.cloud_port}{path}"

    def _post_heartbeat(self, node: SensorNode) -> None:
        payload = {
            "node_id": node.state.node_id,
            "lat": node.state.lat,
            "lon": node.state.lon,
            "status": node.state.status,
            "timestamp": time.time(),
        }
        try:
            requests.post(self._cloud_url("/heartbeat"), json=payload, timeout=5)
            node.state.last_heartbeat = time.time()
            self.store.update_heartbeat(node.state.node_id, node.state.lat, node.state.lon, node.state.status, node.state.last_heartbeat)
        except Exception:
            self.store.update_heartbeat(node.state.node_id, node.state.lat, node.state.lon, "offline", time.time())

    def _send_anomaly(self, node: SensorNode, decision) -> None:
        with decision.audio_path.open("rb") as audio_handle:
            files = {"file": (decision.audio_path.name, audio_handle, "audio/wav")}
            data = {
                "node_id": node.state.node_id,
                "lat": str(node.state.lat),
                "lon": str(node.state.lon),
                "edge_confidence": str(decision.edge_probability),
                "filename": decision.audio_path.name,
            }
            try:
                requests.post(self._cloud_url("/ingest"), files=files, data=data, timeout=15)
            except Exception:
                pass

    def run_forever(self) -> None:
        while True:
            assignments = self.sampler.next_batch([node.state.node_id for node in self.nodes])
            assignment_by_node = {assignment.node_id: assignment for assignment in assignments}
            for node in self.nodes:
                assignment = assignment_by_node[node.state.node_id]
                decision = self.edge_engine.predict(assignment)
                node.state.last_prediction = decision.binary_prediction
                node.state.last_confidence = decision.confidence
                node.state.last_audio = assignment.audio_path.name
                self.store.update_node_prediction(node.state.node_id, decision.binary_prediction, decision.confidence, assignment.audio_path.name)
                if decision.binary_prediction == 1:
                    self._send_anomaly(node, decision)
                if node.state.last_heartbeat == 0.0 or (time.time() - node.state.last_heartbeat) >= self.config.heartbeat_interval_seconds:
                    self._post_heartbeat(node)
            time.sleep(self.config.sample_interval_seconds)


def run_simulator(config: AppConfig, store: LocalStore) -> None:
    EdgeNetworkSimulator(config, store).run_forever()

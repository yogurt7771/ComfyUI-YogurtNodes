import io
import json
import os
import struct
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple
import threading

import requests
from PIL import Image


class ComfyUIClient(object):
    def __init__(
        self,
        server: str,
        workflow: str | Dict[str, Any],
        poll_interval: float = 0.1,
        timeout: Optional[float] = 0,
    ):
        self.SERVER_ADDRESS = self._normalize_server(server)
        self.CLIENT_ID = str(uuid.uuid4())
        self.session: Optional[requests.Session] = None
        self.poll_interval = self._normalize_poll_interval(poll_interval)
        self.timeout = self._normalize_timeout(timeout if timeout > 0 else None)
        self.comfyui_prompt = self._load_workflow(workflow)
        self._title_map = self._build_title_map()

    @staticmethod
    def _normalize_server(server: str) -> str:
        if server.startswith("http://") or server.startswith("https://"):
            return server.rstrip("/")
        return f"http://{server}".rstrip("/")

    @staticmethod
    def _normalize_poll_interval(poll_interval: float) -> float:
        # Keep polling fast enough without hammering the server
        return max(0.05, float(poll_interval))

    @staticmethod
    def _normalize_timeout(timeout: Optional[float]) -> Optional[float]:
        if timeout is None:
            return None
        timeout = float(timeout)
        if timeout <= 0:
            return None
        return timeout

    def _load_workflow(self, workflow: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(workflow, str):
            text = workflow.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid workflow JSON: {exc}") from exc
        elif isinstance(workflow, dict):
            parsed = workflow
        else:
            raise TypeError("workflow must be JSON text or dict")

        if not isinstance(parsed, dict):
            raise RuntimeError("Workflow root must be an object mapping node_id to node data")
        return parsed

    def _build_title_map(self) -> Dict[str, str]:
        title_map: Dict[str, str] = {}
        for key, value in self.comfyui_prompt.items():
            title = value.get("_meta", {}).get("title", "")
            if isinstance(title, str):
                title = title.strip()
                if title:
                    title_map[title] = key
        return title_map

    def _ensure_session(self):
        if self.session is None:
            self.connect()

    def connect(self):
        self.session = requests.Session()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def queue_prompt(self, prompt):
        self._ensure_session()
        payload = {"prompt": prompt, "client_id": self.CLIENT_ID}
        data = json.dumps(payload).encode("utf-8")
        response = self.session.post(
            f"{self.SERVER_ADDRESS}/prompt", data=data, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def get_image(self, filename, subfolder, folder_type):
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(params)
        image_url = f"{self.SERVER_ADDRESS}/view?{url_values}"
        return image_url
        # response = self.session.get(image_url)
        # return response.content

    def get_history(self, prompt_id):
        self._ensure_session()
        response = self.session.get(
            f"{self.SERVER_ADDRESS}/history/{prompt_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def interrupt(self):
        self._ensure_session()
        response = self.session.post(
            f"{self.SERVER_ADDRESS}/api/interrupt", timeout=self.timeout
        )
        return response.status_code == 200

    def queue(self, prompt=None):
        if prompt is None:
            prompt = self.comfyui_prompt
        response = self.queue_prompt(prompt)
        prompt_id = response.get("prompt_id")
        if prompt_id is None:
            raise RuntimeError(json.dumps(response, ensure_ascii=False))
        return prompt_id

    def get_outputs(
        self,
        prompt_id,
        cancel_event: Optional[threading.Event] = None,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ):
        outputs: Dict[str, list] = {}
        poll = self._normalize_poll_interval(
            poll_interval if poll_interval is not None else self.poll_interval
        )
        timeout_seconds = self._normalize_timeout(timeout if timeout is not None else self.timeout)
        start_time = time.time()

        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.interrupt()
                raise RuntimeError("Interrupted")

            history = self.get_history(prompt_id)
            if prompt_id in history and "outputs" in history[prompt_id]:
                break

            if timeout_seconds is not None and time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")

            time.sleep(poll)

        prompt_result = history[prompt_id]
        status = prompt_result.get("status", {})
        if status.get("status_str") != "success":
            raise RuntimeError(json.dumps(status, ensure_ascii=False))

        for node_id, node_output in prompt_result.get("outputs", {}).items():
            output = []
            if "images" in node_output:
                for image in node_output["images"]:
                    image_url = self.get_image(
                        image["filename"], image["subfolder"], image["type"]
                    )
                    output.append({"image": image_url})
            if "text" in node_output:
                for text in node_output["text"]:
                    output.append({"text": text})
            for key, value in node_output.items():
                if key not in {"images", "text"}:
                    output.append({key: value})
            outputs[node_id] = output

        return outputs, prompt_result

    def set_data(self, node: str, key: str, value):
        node_id = self.resolve_node_id(node)
        inputs = self.comfyui_prompt[node_id].setdefault("inputs", {})

        if isinstance(value, Image.Image):
            self._ensure_session()
            folder_name = "temp"

            byte_data = io.BytesIO()
            value.save(byte_data, format="PNG")
            byte_data.seek(0)

            resp = self.session.post(
                f"{self.SERVER_ADDRESS}/upload/image",
                files={"image": ("temp.png", byte_data)},
                data={"subfolder": folder_name},
                timeout=self.timeout,
            )

            resp.raise_for_status()
            resp_json = resp.json()
            inputs[key] = f"{resp_json.get('subfolder', '')}/{resp_json.get('name', '')}"

        else:
            inputs[key] = value

    def set_data_dict(self, node: str, data: Dict[str, Any]):
        for key, value in data.items():
            self.set_data(node, key, value)

    def set_seed(self, seed: int, nodes: Optional[List[str]] = None):
        targets: List[Tuple[str, Dict[str, Any]]] = []
        if nodes:
            for node in nodes:
                node_id = self.resolve_node_id(node)
                targets.append((node_id, self.comfyui_prompt[node_id]))
        else:
            targets = list(self.comfyui_prompt.items())

        for _node_id, node_data in targets:
            inputs = node_data.get("inputs", {})
            for key in inputs:
                if "seed" == key.lower() or "seeds" == key.lower():
                    inputs[key] = seed

    def find_key_by_title(self, target_title: str):
        target_title = target_title.strip()
        if target_title in self._title_map:
            return self._title_map[target_title]

        for key, value in self.comfyui_prompt.items():
            title = value.get("_meta", {}).get("title", "").strip()
            if title == target_title:
                self._title_map[target_title] = key
                return key
        raise RuntimeError(f"Node not found by title: {target_title}")

    def resolve_node_id(self, node: str) -> str:
        node = str(node).strip()
        if node in self.comfyui_prompt:
            return node
        return self.find_key_by_title(node)

    def get_results(
        self,
        prompt_id,
        node_names: Optional[List[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ):
        expected_nodes: List[Tuple[str, str]] = []
        if node_names:
            for node_name in node_names:
                node_id = self.resolve_node_id(node_name)
                expected_nodes.append((node_id, node_name))

        outputs, prompt_result = self.get_outputs(
            prompt_id, cancel_event=cancel_event, timeout=timeout, poll_interval=poll_interval
        )
        results = {}

        if expected_nodes:
            for node_id, node_name in expected_nodes:
                node_outputs = outputs.get(node_id)
                if node_outputs is None:
                    raise RuntimeError(f"Node {node_name}(#{node_id}) not found in outputs")
                results[node_name] = node_outputs
        else:
            results = outputs

        return results, prompt_result

    def run(
        self,
        node_names: Optional[List[str]] = None,
        prompt_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> Dict[str, Any]:
        if prompt_id is None:
            prompt_id = self.queue()

        outputs, prompt_result = self.get_outputs(
            prompt_id,
            cancel_event=cancel_event,
            timeout=timeout,
            poll_interval=poll_interval,
        )

        filtered_outputs = outputs
        if node_names:
            filtered_outputs = {}
            for node_name in node_names:
                node_id = self.resolve_node_id(node_name)
                if node_id not in outputs:
                    raise RuntimeError(f"Node {node_name}(#{node_id}) not found in outputs")
                filtered_outputs[node_id] = outputs[node_id]

        status = prompt_result.get("status", {})
        return {
            "prompt_id": prompt_id,
            "outputs": filtered_outputs,
            "all_outputs": outputs,
            "status": status,
            "workflow": self.comfyui_prompt,
            "node_titles": self._title_map,
            "server": self.SERVER_ADDRESS,
        }

    def generate(
        self,
        node_names: Optional[List[str]] = None,
        prompt_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Backward-compatible wrapper that returns (outputs, status).
        """
        result = self.run(
            node_names=node_names,
            prompt_id=prompt_id,
            cancel_event=cancel_event,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        return result["outputs"], result["status"]

    @staticmethod
    def save_image(image_url: str, save_path: str):
        response = requests.get(image_url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(response.content)

    @staticmethod
    def read_image(image_url: str) -> Image.Image:
        response = requests.get(image_url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise RuntimeError(f"Failed to read image from {image_url}")

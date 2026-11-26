"""Optional continuous OpenAI-compatible completion traffic for a capture window."""

import json
import os
import threading
import time
from urllib.request import Request, urlopen


class Workload:
    def __init__(
        self,
        endpoint,
        prompt,
        *,
        model=None,
        max_tokens=32,
        concurrency=1,
        key_env="OPENAI_API_KEY",
    ):
        if concurrency < 1 or max_tokens < 1:
            raise ValueError("concurrency and max_tokens must be positive")
        self.url = endpoint.rstrip("/")
        self.prompt = prompt
        self.model = model
        self.tokens = max_tokens
        self.concurrency = concurrency
        self.headers = {"Content-Type": "application/json"}
        if os.getenv(key_env):
            self.headers["Authorization"] = "Bearer " + os.environ[key_env]
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.threads = []
        self.requests = []
        self.errors = []

    def complete(self):
        data = {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = Request(
            self.url + "/completions", json.dumps(data).encode(), self.headers
        )
        start = time.time_ns()
        first = None
        tokens = None
        with urlopen(request, timeout=120) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                value = line[5:].strip()
                if value == b"[DONE]":
                    break
                chunk = json.loads(value)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                if (
                    any(c.get("text") for c in chunk.get("choices", []))
                    and first is None
                ):
                    first = time.time_ns()
                if chunk.get("usage"):
                    tokens = chunk["usage"].get("completion_tokens")
        if first is None:
            raise RuntimeError("service returned no completion text")
        return {
            "sent_ns": start,
            "first_content_ns": first,
            "ended_ns": time.time_ns(),
            "output_tokens": tokens,
        }

    def worker(self):
        try:
            while not self.stop_event.is_set():
                result = self.complete()
                with self.condition:
                    self.requests.append(result)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.errors.append(str(exc))
                self.stop_event.set()
                self.condition.notify_all()

    def start(self):
        if self.model is None:
            with urlopen(
                Request(self.url + "/models", headers=self.headers), timeout=30
            ) as response:
                self.model = json.load(response)["data"][0]["id"]
        # Warm up the actual routed service, then keep workers active through export.
        self.complete()
        for _ in range(self.concurrency):
            thread = threading.Thread(target=self.worker, daemon=True)
            self.threads.append(thread)
            thread.start()
        with self.condition:
            if not self.condition.wait_for(
                lambda: self.errors or len(self.requests) >= self.concurrency * 2,
                timeout=180,
            ):
                raise TimeoutError("timed out waiting for inference warmup")
            self.check()

    def check(self):
        if self.errors:
            raise RuntimeError("generation failed: " + self.errors[0])

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        return {
            "model": self.model,
            "concurrency": self.concurrency,
            "max_tokens": self.tokens,
            "requests": self.requests,
            "errors": self.errors,
        }
